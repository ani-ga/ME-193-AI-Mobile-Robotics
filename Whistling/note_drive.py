"""Drive the LEGO Education Double Motor car by whistling / playing notes.

On startup you calibrate which note means which direction:

    1. Press Enter and stay quiet for 1 s  -> measures the room's noise floor
    2. Press Enter and hold your FORWARD note for 1 s
    3. ...same for LEFT, RIGHT and BACKWARD

So whistling can use e.g. C5 for forward, while an instrument can use E4.
After calibration it listens to the microphone, finds the pitch being played,
and drives the car while showing a live spectrogram of the last 5 seconds.

A note counts if the detected pitch is within +/- TOLERANCE Hz (default 20)
of its calibrated frequency; where two ranges overlap, the nearer note wins.
The car keeps doing a command for as long as the note is held and stops
shortly after it ends.

Noise handling: pitch is only searched for inside a band just around your
calibrated notes (a band-pass applied in the FFT), so rumble, hum, hiss and
most overtones are ignored. The quiet-room measurement sets how loud a tone
has to be before it counts.

Calibration is saved to note_calibration.json next to this script;
--use-saved skips calibration and reuses it.

Two computers: at startup it asks whether 1 or 2 computers are streaming
audio. With 2, the "host" laptop is connected to the car and the "remote"
laptop only listens. Each calibrates its own notes, they swap spectrogram
frames over MQTT (both screens show both mics), and either person's whistle
drives the car -- if both play different commands at once, the host wins.

Usage:
    python note_drive.py                  # calibrate, then connect to purple card 0998
    python note_drive.py --use-saved      # reuse the last calibration
    python note_drive.py --no-calibrate   # fixed notes C4/D4/E4/F4 (shift with --octave)
    python note_drive.py --dry-run        # spectrogram + detection, no motors
    python note_drive.py --link host      # 2 computers: this one drives the car
    python note_drive.py --link remote    # 2 computers: this one only streams its mic
"""

import argparse
import base64
import json
import queue
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from matplotlib.animation import FuncAnimation

# ---- Audio / STFT settings ----
SAMPLE_RATE = 44100      # Hz
WINDOW_SIZE = 4096       # samples per analysis window (~93 ms): enough resolution to split notes ~20 Hz apart
FFT_SIZE = 8192          # zero-padded FFT length (~5.4 Hz bins) for a smoother display and finer peaks
HOP_SIZE = 1024          # samples between frames (~23 ms)
HISTORY_SECONDS = 5.0    # how much audio the spectrogram shows
N_COLS = int(HISTORY_SECONDS * SAMPLE_RATE / HOP_SIZE)

# Two-computer streaming
REMOTE_TIMEOUT = 1.0     # s without teammate messages before their commands are ignored

# Wide search range used while calibrating, before we know which notes you'll use.
CAL_FREQ_MIN = 80.0
CAL_FREQ_MAX = 4000.0
NOISE_MARGIN_DB = 12.0   # a tone must be this much louder than the quiet room to count
MIN_VOICED_FRACTION = 0.3  # share of calibration frames that must contain a clear tone

CALIBRATION_FILE = Path(__file__).with_name("note_calibration.json")

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

COMMANDS = ["FORWARD", "LEFT", "RIGHT", "BACKWARD"]
# Used with --no-calibrate: frequencies at octave 4.
DEFAULT_NOTES = {"FORWARD": 261.63, "LEFT": 293.66, "RIGHT": 329.63, "BACKWARD": 349.23}
COMMAND_COLORS = {"FORWARD": "#2ecc71", "LEFT": "#3498db", "RIGHT": "#f1c40f", "BACKWARD": "#e74c3c"}


def freq_to_note(freq):
    """Nearest equal-tempered note name (A4 = 440 Hz), e.g. 'A4'."""
    midi_round = int(np.round(69 + 12 * np.log2(freq / 440.0)))
    return f"{NOTE_NAMES[midi_round % 12]}{midi_round // 12 - 1}"


def parabolic_peak_index(mag, i):
    """Sub-bin peak location via parabolic interpolation around bin i."""
    if i <= 0 or i >= len(mag) - 1:
        return float(i)
    y0, y1, y2 = mag[i - 1], mag[i], mag[i + 1]
    denom = y0 - 2 * y1 + y2
    if denom == 0:
        return float(i)
    # Clamp: if bin i isn't a true local max the parabola can put the peak
    # far away (even at a negative frequency).
    return i + float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))


class PitchDetector:
    """Finds the dominant pitch in one audio frame, or None if the frame is
    silence / noise rather than a clear tone. Only frequencies inside the
    search band are considered, which acts as a band-pass filter."""

    def __init__(self, freq_min, freq_max, min_level_db, min_prominence_db):
        self.window = np.hanning(WINDOW_SIZE)
        # Normalize so a full-scale sine reads about 0 dB.
        self.scale = 2.0 / self.window.sum()
        self.freqs = np.fft.rfftfreq(FFT_SIZE, d=1.0 / SAMPLE_RATE)
        self.bin_hz = self.freqs[1]
        self.min_level_db = min_level_db
        self.min_prominence_db = min_prominence_db
        self.set_band(freq_min, freq_max)

    def set_band(self, freq_min, freq_max):
        self.freq_min, self.freq_max = freq_min, freq_max
        self.band_idx = np.where((self.freqs >= freq_min) & (self.freqs <= freq_max))[0]

    def spectrum_db(self, frame):
        mag = np.abs(np.fft.rfft(frame * self.window, n=FFT_SIZE)) * self.scale
        return 20 * np.log10(mag + 1e-12)

    def band_peak_db(self, spec_db):
        return spec_db[self.band_idx].max()

    def detect(self, spec_db):
        band = spec_db[self.band_idx]
        k = int(np.argmax(band))
        peak_db = band[k]
        # Median of a wide fixed range, so a narrow search band doesn't make
        # every tone look non-prominent.
        floor_db = np.median(spec_db[1:len(spec_db) // 4])
        # A tone must be loud enough AND stand well above the background.
        if peak_db < self.min_level_db or peak_db - floor_db < self.min_prominence_db:
            return None
        freq = parabolic_peak_index(spec_db, self.band_idx[k]) * self.bin_hz

        # Instruments often have a stronger 2nd harmonic than fundamental;
        # if there's a solid peak an octave below, that's the real pitch.
        # (It may land outside the band -- then it's a different, unmapped note.)
        half_bin = int(round(freq / 2 / self.bin_hz))
        if freq / 2 >= CAL_FREQ_MIN:
            lo, hi = half_bin - 2, half_bin + 3
            j = lo + int(np.argmax(spec_db[lo:hi]))
            if spec_db[j] > peak_db - 10 and spec_db[j] - floor_db >= self.min_prominence_db:
                freq = parabolic_peak_index(spec_db, j) * self.bin_hz
        return freq


def frames(signal):
    """Yield successive WINDOW_SIZE frames, HOP_SIZE apart."""
    for start in range(0, len(signal) - WINDOW_SIZE + 1, HOP_SIZE):
        yield signal[start:start + WINDOW_SIZE]


def resolve_device(device):
    """Turn --device (an index or a name substring) into a PyAudio input device index."""
    if device is None or str(device).isdigit():
        return None if device is None else int(device)
    pa = pyaudio.PyAudio()
    try:
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0 and device.lower() in info["name"].lower():
                return i
    finally:
        pa.terminate()
    sys.exit(f"No input device matching '{device}'.")


def record(seconds, device):
    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, input=True,
                         input_device_index=device, frames_per_buffer=HOP_SIZE)
        data = stream.read(int(seconds * SAMPLE_RATE), exception_on_overflow=False)
        stream.stop_stream()
        stream.close()
    finally:
        pa.terminate()
    return np.frombuffer(data, dtype=np.float32)


def calibrate_noise(detector, seconds, device):
    """Record the quiet room and return the level a tone must beat (dBFS)."""
    input("\nStep 1: stay quiet. Press Enter to measure background noise...")
    print(f"  Listening for {seconds:.1f} s...")
    audio = record(seconds, device)
    levels = [detector.band_peak_db(detector.spectrum_db(f)) for f in frames(audio)]
    noise_db = float(np.percentile(levels, 95))
    print(f"  Background noise peaks around {noise_db:.1f} dBFS.")
    return noise_db


def calibrate_note(detector, cmd, seconds, device):
    """Record one held note and return its frequency, or None on failure."""
    input(f"\nPress Enter, then hold your {cmd} note for {seconds:.1f} s...")
    print("  Recording...")
    audio = record(seconds, device)
    pitches = [detector.detect(detector.spectrum_db(f)) for f in frames(audio)]
    voiced = np.array([p for p in pitches if p is not None])
    if len(voiced) < MIN_VOICED_FRACTION * len(pitches):
        print(f"  Only heard a clear tone in {len(voiced)}/{len(pitches)} frames -- too quiet or too noisy.")
        return None
    center = float(np.median(voiced))
    # Ignore octave jumps / stray frames when judging steadiness.
    close = voiced[np.abs(voiced - center) < center * 0.1]
    spread = float(np.percentile(close, 90) - np.percentile(close, 10))
    print(f"  Heard {center:.1f} Hz ({freq_to_note(center)}), wobble {spread:.1f} Hz.")
    return center, spread


def run_calibration(detector, args, commands=COMMANDS):
    noise_db = calibrate_noise(detector, args.record_seconds, args.device)
    detector.min_level_db = max(args.min_level_db, noise_db + NOISE_MARGIN_DB)

    print("\nNow record one note per command. Pick notes at least a couple of "
          f"semitones apart (ranges are +/-{args.tolerance:g} Hz).")
    notes = {}
    for cmd in commands:
        while True:
            result = calibrate_note(detector, cmd, args.record_seconds, args.device)
            if result is None:
                print("  Let's try that again.")
                continue
            center, spread = result
            clash = [c for c, f in notes.items() if abs(f - center) < args.tolerance]
            if clash:
                print(f"  That's within {args.tolerance:g} Hz of {clash[0]} ({notes[clash[0]]:.1f} Hz). "
                      "Pick a more different note.")
                continue
            if spread > args.tolerance:
                print(f"  Warning: the pitch wobbled more than +/-{args.tolerance:g} Hz; "
                      "it may not trigger reliably. (Enter 'r' to redo, or just Enter to keep.)")
                if input("  > ").strip().lower() == "r":
                    continue
            notes[cmd] = center
            break
    return notes, detector.min_level_db


class NoteMapper:
    """Maps a frequency to a command using +/- tolerance ranges, with
    debouncing so single noisy frames don't jerk the car around."""

    def __init__(self, notes, tolerance, confirm_frames, release_frames):
        self.notes = notes  # {command: center Hz}
        self.tolerance = tolerance
        self.confirm_frames = confirm_frames
        self.release_frames = release_frames
        self.active = None       # command currently being executed
        self.candidate = None
        self.candidate_count = 0
        self.silent_count = 0

    def match(self, freq):
        """Command for this frequency, or None if outside every range. In an
        overlap between two ranges, the nearer note wins."""
        if freq is None:
            return None
        best = None
        for cmd, center in self.notes.items():
            dist = abs(freq - center)
            if dist <= self.tolerance and (best is None or dist < best[0]):
                best = (dist, cmd)
        return best[1] if best else None

    def update(self, freq):
        """Feed one frame's pitch; returns the (debounced) active command."""
        cmd = self.match(freq)
        if cmd is None:
            self.candidate, self.candidate_count = None, 0
            self.silent_count += 1
            if self.silent_count >= self.release_frames:
                self.active = None
            return self.active

        self.silent_count = 0
        if cmd == self.active:
            self.candidate, self.candidate_count = None, 0
            return self.active
        if cmd == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate, self.candidate_count = cmd, 1
        if self.candidate_count >= self.confirm_frames:
            self.active = cmd
            self.candidate, self.candidate_count = None, 0
        return self.active


class Car:
    """Thin wrapper around the Double Motor; only sends a BLE command when
    the requested motion actually changes."""

    def __init__(self, card_color, card_serial, speed, turn_speed, dry_run):
        self.speed = speed
        self.turn_speed = turn_speed
        self.motor = None
        self.current = None
        if dry_run:
            print("Dry run: not connecting to the Double Motor.")
            return

        import legoeducation as le

        self.motor = le.DoubleMotor()
        print(f"Connecting to Double Motor (card {card_serial}) over Bluetooth...")
        self.motor.connect(card_color=card_color, card_serial=card_serial)
        if not self.motor.connected:
            sys.exit("Error connecting to Double Motor. Check it's powered on and the card color/serial match.")
        self.motor.movement_set_end_state(le.MOTOR_END_STATE_BRAKE)
        print("Double Motor connected.")

    def command(self, cmd):
        if cmd == self.current:
            return
        self.current = cmd
        print(f"-> {cmd or 'STOP'}")
        if self.motor is None:
            return
        s, t = self.speed, self.turn_speed
        if cmd == "FORWARD":
            self.motor.movement_move_tank(s, s, blocking=False)
        elif cmd == "BACKWARD":
            self.motor.movement_move_tank(-s, -s, blocking=False)
        elif cmd == "LEFT":
            self.motor.movement_move_tank(-t, t, blocking=False)
        elif cmd == "RIGHT":
            self.motor.movement_move_tank(t, -t, blocking=False)
        else:
            self.motor.movement_stop(blocking=False)

    def close(self):
        if self.motor is not None:
            try:
                self.motor.movement_stop()
            finally:
                self.motor.disconnect()
            self.motor = None


def add_common_args(parser):
    parser.add_argument("--card-color", default="PURPLE", help="Double Motor connection card color (default PURPLE)")
    parser.add_argument("--card-serial", default="0998", help="Double Motor connection card serial number (default 0998)")
    parser.add_argument("--speed", type=int, default=40, help="Forward/backward speed percent (default 40)")
    parser.add_argument("--turn-speed", type=int, default=30, help="Turn-in-place speed percent (default 30)")
    parser.add_argument("--use-saved", action="store_true", help="Skip calibration and reuse the saved calibration file")
    parser.add_argument("--no-calibrate", action="store_true", help="Skip calibration and use fixed notes C4/D4/E4/F4")
    parser.add_argument("--octave", type=int, default=0, help="With --no-calibrate: shift the fixed notes by this many octaves (default 0)")
    parser.add_argument("--record-seconds", type=float, default=1.0, help="Length of each calibration recording (default 1.0)")
    parser.add_argument("--tolerance", type=float, default=20.0, help="Accept pitches within +/- this many Hz of each note (default 20)")
    parser.add_argument("--min-level-db", type=float, default=-60.0, help="Ignore tones quieter than this (dBFS, default -60); calibration may raise it")
    parser.add_argument("--min-prominence-db", type=float, default=25.0, help="Tone must stand this many dB above the background spectrum (default 25)")
    parser.add_argument("--confirm-frames", type=int, default=3, help="Frames (~23 ms each) a new note must hold before the car obeys it (default 3)")
    parser.add_argument("--release-frames", type=int, default=8, help="Frames of no command note before the car stops (default 8)")
    parser.add_argument("--device", default=None, help="Input device index or name substring (default: system default mic)")
    parser.add_argument("--dry-run", action="store_true", help="Show spectrogram and detections without connecting to the motors")
    parser.add_argument("--computers", type=int, choices=[1, 2], help="How many computers stream audio (asked at startup if omitted)")
    parser.add_argument("--link", choices=["host", "remote"],
                        help="2 computers: 'host' is connected to the car, 'remote' only streams its mic (implies --computers 2)")
    parser.add_argument("--stream-topic", default=None,
                        help="2 computers: MQTT topic prefix both laptops share (default ME193/Rogers/stream/<card serial>)")


def choose_streaming(args):
    """Ask, unless given on the command line, whether 1 or 2 computers are
    streaming audio and, with 2, whether this one is connected to the car."""
    if args.link:
        args.computers = 2
    while args.computers is None:
        answer = input("How many computers are streaming audio? [1/2] (Enter = 1): ").strip()
        if answer in ("", "1", "2"):
            args.computers = int(answer or 1)
    while args.computers == 2 and args.link is None:
        answer = input("Is THIS computer connected to the robot over Bluetooth? [y/n]: ").strip().lower()
        if answer in ("y", "yes", "n", "no"):
            args.link = "host" if answer.startswith("y") else "remote"
    if args.stream_topic is None:
        args.stream_topic = f"ME193/Rogers/stream/{args.card_serial}"
    if args.computers == 2:
        if args.link == "host":
            print("Two computers: this laptop drives the car; the teammate's whistles are relayed over MQTT.")
        else:
            print("Two computers: this laptop only streams its mic; the teammate's laptop drives the car.")
        print(f"  Both laptops must use the same stream topic: {args.stream_topic}")


def parse_card_color(name):
    import legoeducation as le
    attr = f"LEGO_COLOR_{name.upper()}"
    if not hasattr(le, attr):
        sys.exit(f"Unknown card color '{name}'.")
    return getattr(le, attr)


def load_or_calibrate(detector, args, commands=COMMANDS, cal_file=CALIBRATION_FILE):
    """Return {command: Hz} from the fixed notes, a saved file or a fresh calibration."""
    try:
        if args.no_calibrate:
            return {cmd: DEFAULT_NOTES[cmd] * 2 ** args.octave for cmd in commands}
        if args.use_saved:
            if not cal_file.exists():
                sys.exit(f"No saved calibration at {cal_file}; run without --use-saved first.")
            saved = json.loads(cal_file.read_text())
            missing = [cmd for cmd in commands if cmd not in saved["notes"]]
            if missing:
                sys.exit(f"{cal_file.name} has no note for {', '.join(missing)}; run without --use-saved.")
            detector.min_level_db = saved["min_level_db"]
            return {cmd: saved["notes"][cmd] for cmd in commands}
        notes, min_level_db = run_calibration(detector, args, commands)
        cal_file.write_text(json.dumps({"notes": notes, "min_level_db": min_level_db}, indent=2))
        print(f"\nSaved calibration to {cal_file.name}.")
        return notes
    except OSError as exc:
        sys.exit(f"Could not open the microphone: {exc}")
    except KeyboardInterrupt:
        sys.exit("\nCalibration cancelled.")


def focus_on_notes(detector, notes, args):
    """Band-pass: only look for pitch just around the calibrated notes."""
    search_min = min(notes.values()) - 2 * args.tolerance
    search_max = max(notes.values()) + 2 * args.tolerance
    detector.set_band(max(CAL_FREQ_MIN, search_min), search_max)

    print("\nNote map:")
    for cmd, center in notes.items():
        print(f"  {freq_to_note(center):>3}  {center - args.tolerance:7.1f} - {center + args.tolerance:7.1f} Hz  -> {cmd}")
    print(f"Listening for pitch between {detector.freq_min:.0f} and {detector.freq_max:.0f} Hz, "
          f"louder than {detector.min_level_db:.1f} dBFS.")


def format_pitch(freq):
    if freq is None or np.isnan(freq):
        return "   --"
    return f"{freq:6.1f} Hz ({freq_to_note(freq)})"


class SpectrogramPanel:
    """One scrolling spectrogram with its note bands and detected-pitch line."""

    def __init__(self, fig, ax, title=None):
        self.ax = ax
        if title:
            ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        self.spectrogram = np.full((1, N_COLS), -120.0)
        self.pitch_history = np.full(N_COLS, np.nan)
        self.image = ax.imshow(self.spectrogram, origin="lower", aspect="auto", cmap="magma",
                               extent=[-HISTORY_SECONDS, 0, 0, 1], vmin=-100, vmax=-20, interpolation="bilinear")
        fig.colorbar(self.image, ax=ax, label="Magnitude (dBFS)")
        (self.pitch_line,) = ax.plot(np.linspace(-HISTORY_SECONDS, 0, N_COLS), self.pitch_history,
                                     color="white", lw=2, marker=".", ms=3, zorder=3)
        self.status_text = ax.text(0.01, 0.97, "", transform=ax.transAxes, color="white", fontsize=14,
                                   fontweight="bold", va="top", zorder=4,
                                   bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"))
        self.decor = []   # band shading and labels, redrawn by configure()
        self.bands = {}
        self.config = None

    def configure(self, notes, tolerance, display_max_hz, band, n_bins):
        """Draw the note bands. Called once for this laptop's mic, and again
        for the teammate's whenever their calibration changes."""
        config = (tuple(sorted(notes.items())), tolerance, display_max_hz, tuple(band), n_bins)
        if config == self.config:
            return
        self.config = config
        for artist in self.decor:
            artist.remove()
        self.decor, self.bands = [], {}
        self.spectrogram = np.full((n_bins, N_COLS), -120.0)
        self.image.set_data(self.spectrogram)
        self.image.set_extent([-HISTORY_SECONDS, 0, 0, display_max_hz])
        ax = self.ax
        ax.set_ylim(0, display_max_hz)

        # Grey out frequencies the band-pass ignores.
        self.decor.append(ax.axhspan(0, band[0], color="black", alpha=0.35, lw=0))
        self.decor.append(ax.axhspan(band[1], display_max_hz, color="black", alpha=0.35, lw=0))
        for cmd, center in notes.items():
            color = COMMAND_COLORS.get(cmd, "white")
            self.bands[cmd] = ax.axhspan(center - tolerance, center + tolerance, color=color, alpha=0.15, lw=0)
            label = ax.text(-0.05, center, f"{freq_to_note(center)} {cmd}", color=color, fontsize=9,
                            fontweight="bold", ha="right", va="center")
            self.decor += [self.bands[cmd], label]

    def push(self, cols, pitches):
        """Scroll in new spectrum columns and their detected pitches (NaN = none)."""
        k = min(len(cols), N_COLS)
        if k == 0:
            return
        self.spectrogram = np.roll(self.spectrogram, -k, axis=1)
        self.spectrogram[:, -k:] = np.asarray(cols[-k:]).T
        self.pitch_history = np.roll(self.pitch_history, -k)
        self.pitch_history[-k:] = pitches[-k:]
        self.image.set_data(self.spectrogram)
        # Track loudness so the colors stay readable, but don't amplify silence.
        top = max(self.spectrogram.max(), -50.0)
        self.image.set_clim(top - 70, top)
        self.pitch_line.set_ydata(self.pitch_history)

    def highlight(self, cmd):
        for name, band in self.bands.items():
            band.set_alpha(0.45 if name == cmd else 0.12)

    def set_status(self, text, color="white"):
        self.status_text.set_text(text)
        self.status_text.set_color(color)


class AudioLink:
    """Two-computer mode: sends this laptop's spectrogram frames and detected
    command to the teammate's laptop over MQTT, and receives theirs.

    Spectra are quantized to 0.5 dB steps (one byte per bin) and base64'd,
    which keeps each laptop's stream around 20-30 KB/s."""

    def __init__(self, topic, is_host):
        root = str(Path(__file__).resolve().parent.parent)   # mqttlib lives in the repo root
        if root not in sys.path:
            sys.path.insert(0, root)
        from mqttlib import BROKER, MQTTClient

        me, other = ("host", "remote") if is_host else ("remote", "host")
        self.out_topic = f"{topic}/{me}"
        self.in_topic = f"{topic}/{other}"
        self.inbox = queue.Queue()   # raw payloads from paho's thread
        self.latest = None           # last message from the teammate
        self.last_pitch = None
        self.last_heard = None       # time.monotonic() of that message
        print(f"Connecting to MQTT broker {BROKER} for audio streaming...")
        self.client = MQTTClient()
        self.client.__enter__()
        self.client.subscribe(self.in_topic, lambda topic, payload: self.inbox.put(payload))

    def send(self, cols, pitches, cmd, info):
        spec = np.clip(np.round((np.asarray(cols, dtype=float).reshape(len(cols), -1) + 127.5) * 2), 0, 255)
        msg = dict(info, cmd=cmd, k=len(cols),
                   spec=base64.b64encode(spec.astype(np.uint8).tobytes()).decode("ascii"),
                   pitch=[None if np.isnan(p) else round(float(p), 1) for p in pitches])
        self.client.publish(self.out_topic, json.dumps(msg))

    def receive(self):
        """Decode every teammate message that arrived since the last call."""
        msgs = []
        while True:
            try:
                payload = self.inbox.get_nowait()
            except queue.Empty:
                break
            try:
                msg = json.loads(payload)
                spec = np.frombuffer(base64.b64decode(msg["spec"]), dtype=np.uint8)
                msg["cols"] = spec.reshape(msg["k"], msg["n_bins"]) / 2.0 - 127.5
                msg["pitch"] = [np.nan if p is None else p for p in msg["pitch"]]
            except (ValueError, KeyError, TypeError):
                continue   # garbled, or something else published on the topic
            msgs.append(msg)
            self.latest, self.last_heard = msg, time.monotonic()
            if msg["k"]:
                self.last_pitch = msg["pitch"][-1]
        return msgs

    @property
    def connected(self):
        return self.last_heard is not None and time.monotonic() - self.last_heard < REMOTE_TIMEOUT

    def teammate_cmd(self):
        return self.latest["cmd"] if self.connected else None

    def close(self):
        self.client.__exit__(None, None, None)


def run_live(detector, notes, car, args, game=None):
    """Listen, drive and show the live spectrogram until the window is closed.

    `game` is optional (see world_cup.py): game.step(heard_cmd) is called on
    every screen update and returns the command the car should actually do,
    and game.status() returns (text, color) for an extra status box.

    With args.computers == 2 a second panel shows the teammate's mic. On the
    host the car obeys this laptop's note, or else the teammate's; the remote
    never drives and just shows what the host's car is doing."""
    mapper = NoteMapper(notes, args.tolerance, args.confirm_frames, args.release_frames)
    display_max_hz = max(notes.values()) * 2.5
    disp_bins = np.where(detector.freqs <= display_max_hz)[0]
    two = getattr(args, "computers", 1) == 2
    is_host = not two or args.link == "host"

    link = None
    if two:
        try:
            link = AudioLink(args.stream_topic, is_host)
        except OSError as exc:
            car.close()
            if game is not None:
                game.close()
            sys.exit(f"Could not connect to the MQTT broker for streaming: {exc}")
    info = {"notes": notes, "tol": args.tolerance, "max_hz": display_max_hz,
            "band": [detector.freq_min, detector.freq_max], "n_bins": len(disp_bins)}

    # ---- Audio buffers ----
    audio_queue = queue.Queue()
    pending = np.zeros(0, dtype=np.float32)   # samples not yet consumed by a hop
    window_buf = np.zeros(WINDOW_SIZE, dtype=np.float32)
    state = {"freq": None, "cmd": None}

    def audio_callback(in_data, frame_count, time_info, status):
        if status:
            print(f"Audio status flag {status}", file=sys.stderr)
        audio_queue.put(np.frombuffer(in_data, dtype=np.float32).copy())
        return None, pyaudio.paContinue

    # ---- Figure ----
    fig, axes = plt.subplots(2 if two else 1, 1, figsize=(12, 9 if two else 6), squeeze=False)
    local_title, mate_title = None, None
    if two:
        local_title = "This computer (drives the car)" if is_host else "This computer (remote mic)"
        mate_title = "Teammate (remote mic)" if is_host else "Teammate (drives the car)"
    local = SpectrogramPanel(fig, axes[0, 0], local_title)
    local.configure(notes, args.tolerance, display_max_hz, info["band"], len(disp_bins))
    mate = None
    if two:
        mate = SpectrogramPanel(fig, axes[1, 0], mate_title)
        mate.set_status(f"Waiting for teammate on {link.in_topic}...")
    game_text = None
    if game is not None or not is_host:   # the remote shows the host's game status
        game_text = axes[0, 0].text(0.99, 0.97, "", transform=axes[0, 0].transAxes, color="white", fontsize=14,
                                    fontweight="bold", va="top", ha="right", zorder=4,
                                    bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"))

    def update(_):
        nonlocal pending
        chunks = []
        while True:
            try:
                chunks.append(audio_queue.get_nowait())
            except queue.Empty:
                break
        if chunks:
            pending = np.concatenate([pending, *chunks])

        new_cols = []
        new_pitches = []
        while len(pending) >= HOP_SIZE:
            window_buf[:-HOP_SIZE] = window_buf[HOP_SIZE:]
            window_buf[-HOP_SIZE:] = pending[:HOP_SIZE]
            pending = pending[HOP_SIZE:]
            spec_db = detector.spectrum_db(window_buf)
            freq = detector.detect(spec_db)
            state["freq"] = freq
            state["cmd"] = mapper.update(freq)
            new_cols.append(spec_db[disp_bins])
            new_pitches.append(np.nan if freq is None else freq)
        local.push(new_cols, new_pitches)
        freq, cmd = state["freq"], state["cmd"]

        mate_cmd = None
        if link is not None:
            for msg in link.receive():
                mate.configure(msg["notes"], msg["tol"], msg["max_hz"], msg["band"], msg["n_bins"])
                mate.push(msg["cols"], msg["pitch"])
            mate_cmd = link.teammate_cmd()

        game_status = None
        if is_host:
            heard = cmd or mate_cmd   # this laptop wins when both play a note
            drive = game.step(heard) if game is not None else heard
            car.command(drive)
            car_text = drive or "STOP"
            if game is not None:
                game_status = game.status()
        else:
            host = link.latest if link.connected else None
            drive = host.get("drive") if host else None
            car_text = (drive or "STOP") if host else "(host offline)"
            if host and host.get("status"):
                game_status = tuple(host["status"])

        if link is not None:
            link.send(new_cols, new_pitches, cmd,
                      dict(info, drive=drive if is_host else None, status=game_status))
            if link.connected:
                mate.highlight(mate_cmd)
                mate.set_status(f"Teammate heard: {format_pitch(link.last_pitch)}    -> {mate_cmd or '--'}",
                                COMMAND_COLORS.get(mate_cmd, "white"))
            else:
                mate.highlight(None)
                mate.set_status(f"Waiting for teammate on {link.in_topic}...")

        local.highlight(cmd)
        local.set_status(f"Heard: {format_pitch(freq)}    Car: {car_text}", COMMAND_COLORS.get(drive, "white"))
        if game_text is not None:
            text, color = game_status or ("", "white")
            game_text.set_text(text)
            game_text.set_color(color)
            game_text.set_visible(bool(text))
        return []

    pa = pyaudio.PyAudio()
    stream = None
    try:
        stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, input=True,
                         input_device_index=args.device, frames_per_buffer=HOP_SIZE,
                         stream_callback=audio_callback)
        anim = FuncAnimation(fig, update, interval=40, blit=False, cache_frame_data=False)
        fig.tight_layout()
        print("Listening... close the plot window (or Ctrl+C) to stop.")
        plt.show()
        del anim
    except OSError as exc:
        sys.exit(f"Could not open the microphone: {exc}")
    except KeyboardInterrupt:
        pass
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        pa.terminate()
        car.close()
        if game is not None:
            game.close()
        if link is not None:
            link.close()
        print('Closing...')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    args = parser.parse_args()
    choose_streaming(args)

    card_color = parse_card_color(args.card_color) if args.card_color else None
    args.device = resolve_device(args.device)

    detector = PitchDetector(CAL_FREQ_MIN, CAL_FREQ_MAX, args.min_level_db, args.min_prominence_db)
    notes = load_or_calibrate(detector, args)
    focus_on_notes(detector, notes, args)

    # The remote laptop never connects to the car; the host drives it.
    car = Car(card_color, args.card_serial, args.speed, args.turn_speed, args.dry_run or args.link == "remote")
    run_live(detector, notes, car, args)


if __name__ == "__main__":
    main()
