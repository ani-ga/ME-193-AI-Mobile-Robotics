"""World Cup: whistle-drive the car as the "ball" or the "goalie".

Driving works exactly like note_drive.py (same calibration, pitch detection,
noise filtering and live spectrogram). This adds the game around it:

    WAITING   the car ignores whistles until "start" arrives on ME193/Rogers
    PLAYING   whistle to drive
    WON/LOST  motors stop, whistles are ignored and a song plays
              until the next "start", which begins a new round

MQTT ("start" always comes from ME193/Rogers):
  - Results are JSON on ME193/Rogers:  {"event": "goal", "team": "cucurella"}
      goal = that team's ball scored,  fail = that team's ball was caught
    Only messages whose "team" is the opponent (--team) count; others are
    printed and ignored. Our ball publishes the same JSON with --my-team.
  - Plain-text "goal" / "fail" on ME193/<opponent> are also accepted.

Ball (calibrates a 5th note, GOAL):
  - light sensor sees the goalie up close -> publish "fail", death song
  - hold the GOAL note for GOAL_HOLD_SECONDS -> publish "goal", success song
Goalie:
  - opponent's "fail" -> success song
  - opponent's "goal" -> death song
With two computers the remote laptop plays the same song as the host.

The light sensor counts the ball as caught when its reflection reading rises
--catch-delta above the baseline measured at startup for --catch-frames
screen updates in a row (anything close in front of it reflects its light).

Calibration is saved to world_cup_calibration.json next to this script.

Usage:
    python world_cup.py --role ball              # calibrate, connect car + sensor, wait for start
    python world_cup.py --role goalie --use-saved
    python world_cup.py --role ball --dry-run    # no motors or sensor
    python world_cup.py --role goalie --team cucurella   # opponent is team cucurella
    python world_cup.py --send start             # publish a test message and exit
    python world_cup.py --send goal --team cucurella     # pretend cucurella's ball scored
"""

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pyaudio

import note_drive as nd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mqttlib import BROKER, MQTTClient  # noqa: E402  (lives in the repo root)

START_TOPIC = "ME193/Rogers"     # "start" and JSON results go here
DEFAULT_MY_TEAM = "Ryan"         # our name in the JSON "team" field; change with --my-team
MSG_START = "start"
MSG_BALL_CAUGHT = "fail"   # the ball was caught: the goalie wins
MSG_BALL_SCORED = "goal"   # the ball scored: the goalie loses

GOAL_HOLD_SECONDS = 1.0           # hold GOAL this long so a stray note can't end the game
SENSOR_BASELINE_SECONDS = 1.0

CALIBRATION_FILE = Path(__file__).with_name("world_cup_calibration.json")

nd.DEFAULT_NOTES["GOAL"] = 392.00          # G4, used with --no-calibrate
nd.COMMAND_COLORS["GOAL"] = "#9b59b6"

# (frequency Hz, seconds); 0 Hz is a rest.
SUCCESS_SONG = [(523.25, 0.15), (659.25, 0.15), (783.99, 0.15), (1046.50, 0.4),
                (0, 0.1), (783.99, 0.15), (1046.50, 0.6)]
DEATH_SONG = [(493.88, 0.45), (466.16, 0.45), (440.00, 0.45), (415.30, 1.3)]  # wah wah wah waaah
SONGS = {"success": SUCCESS_SONG, "death": DEATH_SONG}   # names the host streams to the remote

GAME_COLORS = {"WAITING": "white", "PLAYING": "#2ecc71", "WON": "#f1c40f", "LOST": "#e74c3c"}


def play_song(song, volume=0.3):
    """Play a list of (Hz, seconds) notes through the speakers without
    blocking the live plot."""
    def worker():
        parts = []
        for freq, seconds in song:
            t = np.arange(int(seconds * nd.SAMPLE_RATE)) / nd.SAMPLE_RATE
            tone = np.sin(2 * np.pi * freq * t)
            fade = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.01)  # 10 ms fades, no clicks
            parts.append(tone * fade)
        audio = (volume * np.concatenate(parts)).astype(np.float32)
        pa = pyaudio.PyAudio()
        try:
            out = pa.open(format=pyaudio.paFloat32, channels=1, rate=nd.SAMPLE_RATE, output=True)
            out.write(audio.tobytes())
            out.stop_stream()
            out.close()
        except OSError as exc:
            print(f"Could not play song: {exc}", file=sys.stderr)
        finally:
            pa.terminate()

    threading.Thread(target=worker, daemon=True).start()


class LightSensor:
    """Color Sensor used as a trip-wire: 'tripped' when the reflection reading
    stays well above the baseline measured at startup."""

    def __init__(self, card_color, card_serial, delta, frames):
        import legoeducation as le

        self.delta = delta
        self.frames = frames
        self.count = 0
        self.last = None
        self.sensor = le.ColorSensor()
        print(f"Connecting to Color Sensor (card {card_serial}) over Bluetooth...")
        self.sensor.connect(card_color=card_color, card_serial=card_serial)
        if not self.sensor.connected:
            sys.exit("Error connecting to Color Sensor. Check it's powered on and the card color/serial match.")

        print(f"Measuring light sensor baseline for {SENSOR_BASELINE_SECONDS:.0f} s -- keep the area in front of it clear...")
        readings = []
        end = time.monotonic() + SENSOR_BASELINE_SECONDS
        while time.monotonic() < end:
            value = self.reflection()
            if value is not None:
                readings.append(value)
            time.sleep(0.05)
        if not readings:
            sys.exit("The Color Sensor didn't send any readings.")
        self.baseline = float(np.median(readings))
        print(f"Light sensor baseline reflection {self.baseline:.0f}; caught above {self.threshold:.0f}.")

    @property
    def threshold(self):
        return self.baseline + self.delta

    def reflection(self):
        value = float(self.sensor.sensor.reflection)
        return None if np.isnan(value) else value   # NaN until the first reading arrives

    def tripped(self):
        self.last = self.reflection()
        if self.last is not None and self.last > self.threshold:
            self.count += 1
        else:
            self.count = 0
        return self.count >= self.frames

    def close(self):
        if self.sensor is not None:
            self.sensor.disconnect()
            self.sensor = None


def result_json(event, team):
    return json.dumps({"event": event, "team": team})


def parse_result(msg):
    """(event, team) from '{"event": "goal", "team": "cucurella"}', or
    (None, None) if msg isn't a result in that form."""
    try:
        data = json.loads(msg)
    except ValueError:
        return None, None
    if not isinstance(data, dict) or "event" not in data:
        return None, None
    return str(data["event"]).strip().lower(), str(data.get("team", "")).strip().lower()


class Game:
    """World Cup rules on top of note_drive.run_live(). step() is called about
    25 times a second with the command that was heard and returns what the car
    should actually do (None = stop)."""

    def __init__(self, role, mqtt, sensor, opponent, my_team):
        self.role = role
        self.mqtt = mqtt
        self.sensor = sensor
        self.opponent = opponent
        self.my_team = my_team
        team_topic = f"ME193/{opponent}"   # plain-text results from the opponent
        self.state = "WAITING"
        self.reason = f"waiting for '{MSG_START}'"
        self.goal_since = None
        self.song = [0, None]   # [songs played, last song name]; streamed so the remote plays it too
        # MQTT messages arrive on paho's thread; step() handles them on the plot's thread.
        self.messages = queue.Queue()
        for topic in {START_TOPIC, team_topic}:
            mqtt.subscribe(topic, lambda topic, payload: self.messages.put((topic, payload.strip().lower())))

    def step(self, heard):
        while True:
            try:
                self.on_message(*self.messages.get_nowait())
            except queue.Empty:
                break
        # Read the sensor in every state so its value is on screen before the start.
        caught = self.sensor is not None and self.sensor.tripped()
        if self.state != "PLAYING":
            return None

        if self.role == "ball":
            if caught:
                self.finish(False, "caught by the goalie", publish=MSG_BALL_CAUGHT)
                return None
            if heard == "GOAL":
                if self.goal_since is None:
                    self.goal_since = time.monotonic()
                if time.monotonic() - self.goal_since >= GOAL_HOLD_SECONDS:
                    self.finish(True, "GOAL!", publish=MSG_BALL_SCORED)
                return None   # stay still while the GOAL note is held
            self.goal_since = None
        elif heard == "GOAL":
            return None
        return heard

    def on_message(self, topic, msg):
        print(f"MQTT [{topic}] {msg}")
        if msg == MSG_START and topic == START_TOPIC:
            # Every "start" begins a fresh round, even after a win or loss.
            restart = self.state != "WAITING"
            self.state, self.reason = "PLAYING", "go!" if not restart else "new round -- go!"
            self.goal_since = None
            if self.sensor is not None:
                self.sensor.count = 0
            print("New round! Whistle to drive." if restart else "Start! Whistle to drive.")
            return

        event, team = parse_result(msg)
        if event is None:
            if topic == START_TOPIC:
                return   # plain text on the shared topic has no team -- can't tell who sent it
            event, team = msg, self.opponent.lower()
        if team != self.opponent.lower():
            print(f"  (ignored: from team '{team}', not our opponent '{self.opponent}')")
            return
        if self.role == "goalie" and self.state in ("WAITING", "PLAYING"):
            if event == MSG_BALL_CAUGHT:
                self.finish(True, f"caught {self.opponent}'s ball")
            elif event == MSG_BALL_SCORED:
                self.finish(False, f"{self.opponent} scored")

    def finish(self, won, reason, publish=None):
        self.state = "WON" if won else "LOST"
        self.reason = reason
        print(f"\n*** {self.role.upper()} {self.state}: {reason} ***")
        if publish:
            payload = result_json(publish, self.my_team)
            self.mqtt.publish(START_TOPIC, payload)
            print(f"Published {payload} to {START_TOPIC}.")
        name = "success" if won else "death"
        self.song = [self.song[0] + 1, name]
        play_song(SONGS[name])

    def status(self):
        text = f"{self.role.upper()} | {self.state} | {self.reason}"
        if self.sensor is not None and self.sensor.last is not None:
            text += f"\nlight {self.sensor.last:.0f} (caught > {self.sensor.threshold:.0f})"
        return text, GAME_COLORS[self.state]

    def close(self):
        if self.sensor is not None:
            self.sensor.close()


def ask_opponent(args):
    """The opponent's team name, from --team or asked at startup."""
    while not args.team:
        args.team = input("Opponent's team name (as it appears in their \"team\" field): ").strip()
    return args.team


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", choices=["ball", "goalie"], help="Which side you're playing")
    parser.add_argument("--team", default=None,
                        help="Opponent's team name: only their goal/fail results count (asked at startup if omitted)")
    parser.add_argument("--my-team", default=DEFAULT_MY_TEAM,
                        help=f"Our team name, sent in the JSON results we publish (default {DEFAULT_MY_TEAM})")
    parser.add_argument("--send", metavar="MESSAGE",
                        help=f"Just publish MESSAGE to {START_TOPIC} and exit (for testing). "
                             "'goal' / 'fail' are sent as JSON from --team, as if the opponent sent them")
    parser.add_argument("--sensor-card-color", default="PURPLE", help="Color Sensor connection card color (default PURPLE)")
    parser.add_argument("--sensor-card-serial", default="0998", help="Color Sensor connection card serial number (default 0998)")
    parser.add_argument("--catch-delta", type=float, default=15.0,
                        help="Reflection rise above baseline that means the goalie is close (default 15)")
    parser.add_argument("--catch-frames", type=int, default=3,
                        help="Screen updates (~40 ms each) the reflection must stay high (default 3)")
    parser.add_argument("--no-sensor", action="store_true", help="Ball without the light sensor (it can only score)")
    nd.add_common_args(parser)
    args = parser.parse_args()

    if args.send:
        payload = args.send
        if payload.strip().lower() in (MSG_BALL_SCORED, MSG_BALL_CAUGHT):
            payload = result_json(payload.strip().lower(), ask_opponent(args))
        with MQTTClient() as mqtt:
            mqtt.publish(START_TOPIC, payload)
            time.sleep(1)  # give the message time to reach the broker before disconnecting
        print(f"Published {payload} to {START_TOPIC}.")
        return
    if args.role is None:
        parser.error("--role ball or --role goalie is required")
    nd.choose_streaming(args)
    if args.link != "remote":   # the remote doesn't run the game
        ask_opponent(args)

    commands = nd.COMMANDS + (["GOAL"] if args.role == "ball" else [])
    card_color = nd.parse_card_color(args.card_color) if args.card_color else None
    args.device = nd.resolve_device(args.device)

    detector = nd.PitchDetector(nd.CAL_FREQ_MIN, nd.CAL_FREQ_MAX, args.min_level_db, args.min_prominence_db)
    notes = nd.load_or_calibrate(detector, args, commands, CALIBRATION_FILE)
    nd.focus_on_notes(detector, notes, args)

    if args.link == "remote":
        # Just a second mic: the host laptop runs the car, sensor and game;
        # its game status shows up on this screen and its songs play here too.
        car = nd.Car(card_color, args.card_serial, args.speed, args.turn_speed, dry_run=True)
        nd.run_live(detector, notes, car, args, on_host_song=lambda name: play_song(SONGS[name]))
        return

    car = nd.Car(card_color, args.card_serial, args.speed, args.turn_speed, args.dry_run)
    sensor = None
    if args.role == "ball" and not (args.dry_run or args.no_sensor):
        sensor = LightSensor(nd.parse_card_color(args.sensor_card_color), args.sensor_card_serial,
                             args.catch_delta, args.catch_frames)

    print(f"Connecting to MQTT broker {BROKER}...")
    with MQTTClient() as mqtt:
        game = Game(args.role, mqtt, sensor, args.team, args.my_team)
        print(f"Role: {args.role} (team {args.my_team}) vs {args.team}. "
              f"Waiting for '{MSG_START}' on {START_TOPIC}...")
        nd.run_live(detector, notes, car, args, game)


if __name__ == "__main__":
    main()
