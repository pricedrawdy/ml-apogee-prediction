"""
flight_demo.py — Raspberry Pi rocket flight demo
================================================

Replays a real RocketPy simulated flight (noisy sensor data) in real-time,
runs ML apogee prediction at burnout, and prints a summary.

Usage
-----
    python flight_demo.py                          # MLP model, flight #0
    python flight_demo.py --model rf               # Random Forest
    python flight_demo.py --model lr               # Linear Regression
    python flight_demo.py --flight 5               # Different flight from dataset
    python flight_demo.py --no-audio               # Silent / headless (SSH)
    python flight_demo.py --serial /dev/ttyUSB0    # Arduino serial feed

Controls
--------
    Ctrl-C   abort at any time
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ── Optional imports (graceful fallback) ─────────────────────────────────────
try:
    import serial as pyserial
    _SERIAL_OK = True
except ImportError:
    _SERIAL_OK = False

try:
    import pygame
    _PYGAME_OK = True
except ImportError:
    _PYGAME_OK = False

# ── Local imports ─────────────────────────────────────────────────────────────
_DEPLOY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DEPLOY_DIR))

from fake_flight_generator import DemoFlight, FlightProfile, create_demo_flight, TIMESTEP
from inference_engine import InferenceEngine

# ── ANSI colour helpers ───────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"
    ORANGE = "\033[38;5;214m"
    GRAY   = "\033[90m"

def banner(msg: str, colour: str = C.CYAN) -> None:
    w = 60
    print(f"\n{colour}{C.BOLD}{'=' * w}")
    print(f"  {msg}")
    print(f"{'=' * w}{C.RESET}\n")

def status(label: str, value: str, colour: str = C.WHITE) -> None:
    print(f"  {C.GRAY}{label:<26}{C.RESET}{colour}{value}{C.RESET}")


# ── Sound ─────────────────────────────────────────────────────────────────────
SOUNDS_DIR = _DEPLOY_DIR / "sounds"

def _init_audio() -> bool:
    if not _PYGAME_OK:
        print(f"{C.YELLOW}[Audio] pygame not available — running silently.{C.RESET}")
        return False
    try:
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        return True
    except Exception as e:
        print(f"{C.YELLOW}[Audio] Could not init audio: {e}{C.RESET}")
        return False


def _play(sound_file: str, audio_ok: bool, loop: bool = False):
    if not audio_ok or not _PYGAME_OK:
        return None
    path = SOUNDS_DIR / sound_file
    if not path.exists():
        print(f"{C.YELLOW}[Audio] Missing: {path.name}  (run: python download_sounds.py){C.RESET}")
        return None
    try:
        sound = pygame.mixer.Sound(str(path))
        loops = -1 if loop else 0
        return sound.play(loops=loops)
    except Exception as e:
        print(f"{C.YELLOW}[Audio] Playback error ({sound_file}): {e}{C.RESET}")
        return None


def _stop(ch) -> None:
    if ch is not None and _PYGAME_OK:
        try:
            ch.stop()
        except Exception:
            pass


# ── Countdown ─────────────────────────────────────────────────────────────────
def countdown(seconds: int, audio_ok: bool) -> None:
    banner("PRE-LAUNCH COUNTDOWN", C.CYAN)
    _play("igniter_arm.wav", audio_ok)
    for i in range(seconds, 0, -1):
        colour = C.RED if i <= 3 else C.YELLOW if i <= 5 else C.WHITE
        print(f"  {colour}{C.BOLD}T-{i:02d}{C.RESET}", flush=True)
        time.sleep(1.0)
    print()


# ── Telemetry display ─────────────────────────────────────────────────────────
def _velocity_bar(speed_ms: float, max_ms: float = 250.0) -> str:
    """Horizontal bar proportional to speed."""
    frac    = min(abs(speed_ms) / max_ms, 1.0)
    bar_len = int(frac * 30)
    bar     = C.GREEN + "#" * bar_len + C.GRAY + "-" * (30 - bar_len) + C.RESET
    return bar

def print_telemetry(step: dict) -> None:
    t    = step["t"]
    alt  = step["altitude"]
    vv   = step["v_vel"]     # negative = ascending in RocketPy convention
    vacc = step["v_acc"]
    mach = step["mach"]
    speed_ms = abs(vv)

    print(
        f"\r  {C.CYAN}T+{t:5.2f}s{C.RESET}  "
        f"Alt:{C.YELLOW}{alt:7.1f}m{C.RESET}  "
        f"Speed:{C.GREEN}{speed_ms:6.1f}m/s{C.RESET}  "
        f"Acc:{C.ORANGE}{vacc:+6.2f}m/s2{C.RESET}  "
        f"M:{mach:.3f}  "
        f"{_velocity_bar(speed_ms)}",
        end="", flush=True,
    )


# ── Burnout detection ─────────────────────────────────────────────────────────
def detect_burnout(history: list[dict], burn_time: float) -> bool:
    """
    Trigger inference at t=5.5 s — the midpoint of the burnout inference window
    (5.0–8.0 s) used during training.

    Using a fixed time avoids false positives from drag-induced acceleration
    dips during powered flight, and guarantees the feature window is always
    within the distribution the models were trained on.
    """
    if not history:
        return False
    return history[-1]["t"] >= 5.5


# ── Arduino serial reader ─────────────────────────────────────────────────────
class ArduinoReader:
    """
    Reads CSV telemetry from an Arduino over USB serial.

    Expected line format (25 ms intervals):
        t,altitude,v_vel,v_acc,t_vel,h_vel,pitch,dynp,mach,pressure
    Comment lines start with #.

    Note: The Arduino sketch (arduino/fake_flight_sensor/fake_flight_sensor.ino)
    outputs data in the same sign convention as the Pi generator.
    """
    FIELDS = ("t", "altitude", "v_vel", "v_acc", "t_vel",
              "h_vel", "pitch", "dynp", "mach", "pressure")

    def __init__(self, port: str, baud: int):
        if not _SERIAL_OK:
            raise ImportError("pyserial not installed: pip install pyserial")
        self._ser = pyserial.Serial(port, baud, timeout=2.0)
        print(f"[Serial] Connected: {port} @ {baud} baud")
        time.sleep(2.0)
        self._ser.flushInput()

    def __iter__(self):
        for raw in self._ser:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < len(self.FIELDS):
                continue
            try:
                step = {k: float(v) for k, v in zip(self.FIELDS, parts)}
                step["step_idx"] = -1
                yield step
            except ValueError:
                continue

    def close(self):
        self._ser.close()


# ── Build inference window from rolling history ───────────────────────────────
def _window_from_history(history: list[dict]) -> np.ndarray:
    """
    Build the 1000-element feature vector from the rolling telemetry history.
    Uses the most recent WINDOW_SIZE (100) steps.
    """
    from fake_flight_generator import WINDOW_SIZE
    win = history[-WINDOW_SIZE:]
    pad = WINDOW_SIZE - len(win)

    def _col(key: str) -> np.ndarray:
        arr = np.array([h[key] for h in win], dtype=float)
        if pad:
            arr = np.concatenate([np.zeros(pad), arr])
        return arr

    w_v_vel  = _col("v_vel")
    w_v_acc  = _col("v_acc")
    w_t_vel  = _col("t_vel")
    w_h_vel  = _col("h_vel")
    w_pitch  = _col("pitch")
    w_dynp   = _col("dynp")
    w_mach   = _col("mach")
    w_press  = _col("pressure")
    w_alt    = _col("altitude")

    relative_alt = w_alt - w_alt[0]
    vel_squared  = w_t_vel ** 2

    return np.concatenate([
        w_v_vel, w_v_acc, w_t_vel, w_h_vel, w_pitch,
        w_dynp, w_mach, w_press, relative_alt, vel_squared,
    ])


# ── Main demo ─────────────────────────────────────────────────────────────────
def run_demo(
    model_type:   str       = "mlp",
    flight_index: int       = 0,
    no_audio:     bool      = False,
    serial_port:  str | None = None,
    baud:         int       = 115200,
) -> None:

    audio_ok = (not no_audio) and _init_audio()

    # ── Load flight data ───────────────────────────────────────────────────
    if serial_port:
        # Arduino mode — use a reference flight for metadata only
        profile = FlightProfile(flight_index=flight_index)
        demo = create_demo_flight(profile)
        source = ArduinoReader(serial_port, baud)
        use_serial = True
    else:
        demo = DemoFlight(flight_index=flight_index)
        source = demo.stream()
        use_serial = False

    true_apogee = demo.true_apogee_m
    burn_time   = demo.burn_time

    banner("ROCKET APOGEE PREDICTOR  --  DEMO FLIGHT", C.CYAN)
    status("Model",           model_type.upper())
    status("Flight index",    str(flight_index))
    status("True Apogee",     f"{true_apogee:.1f} m  ({true_apogee * 3.28084:.0f} ft)")
    status("Motor Burn Time", f"{burn_time:.2f} s")
    status("Audio",           "ON" if audio_ok else "OFF")
    status("Data source",     f"Arduino ({serial_port})" if use_serial else "Dataset replay")
    print()

    # ── Load ML model ─────────────────────────────────────────────────────
    engine = InferenceEngine()
    engine.load(model_type)

    # ── Countdown ─────────────────────────────────────────────────────────
    countdown(10, audio_ok)

    # ── Liftoff ───────────────────────────────────────────────────────────
    banner("IGNITION -- LIFTOFF!", C.RED)
    motor_ch = _play("motor_burn.wav", audio_ok, loop=True)
    _play("ignition.wav", audio_ok)

    history: list[dict]         = []
    prediction_done:  bool      = False
    prediction_result           = None
    burnout_announced: bool      = False

    try:
        for step in source:
            history.append(step)
            print_telemetry(step)

            # Simulate real-time pacing for dataset replay
            if not use_serial:
                time.sleep(TIMESTEP)

            # ── Burnout detection ──────────────────────────────────────────
            if not burnout_announced and detect_burnout(history, burn_time):
                _stop(motor_ch)
                _play("burnout.wav", audio_ok)
                print()  # close telemetry line
                banner("BURNOUT DETECTED", C.ORANGE)
                status("T+ flight",   f"{step['t']:.2f} s")
                status("Altitude",    f"{step['altitude']:.1f} m  ({step['altitude']*3.28084:.0f} ft)")
                status("Speed",       f"{abs(step['v_vel']):.1f} m/s")
                burnout_announced = True

            # ── ML Inference — once, right at burnout ──────────────────────
            if burnout_announced and not prediction_done:
                feat = _window_from_history(history)
                print(f"\n{C.CYAN}[ML] Running {model_type.upper()} inference ...{C.RESET}", flush=True)
                result = engine.predict(feat)
                prediction_done   = True
                prediction_result = result

                banner("APOGEE PREDICTION", C.GREEN)
                status("Predicted",  f"{result.apogee_m:.1f} m  ({result.apogee_ft:.0f} ft)", C.GREEN)
                status("Model",      result.model_type.upper())
                status("Latency",    f"{result.elapsed_ms:.1f} ms")
                print()
                _play("prediction_beep.wav", audio_ok)

            # ── Apogee reached ─────────────────────────────────────────────
            t_now = step["t"]
            speed = abs(step["v_vel"])
            if t_now > burn_time + 2.0 and speed < 5.0:
                print()
                _stop(motor_ch)
                _play("apogee_beep.wav", audio_ok)
                break

    except KeyboardInterrupt:
        print(f"\n{C.RED}Aborted by user.{C.RESET}")
        _stop(motor_ch)

    finally:
        if use_serial:
            source.close()

    # ── Summary ───────────────────────────────────────────────────────────
    banner("FLIGHT SUMMARY", C.CYAN)
    status("True Apogee",
           f"{true_apogee:.1f} m  ({true_apogee * 3.28084:.0f} ft)", C.YELLOW)

    if prediction_result:
        err_m  = prediction_result.apogee_m - true_apogee
        err_pc = (err_m / true_apogee) * 100.0
        col = C.GREEN if abs(err_pc) < 2.0 else C.ORANGE if abs(err_pc) < 5.0 else C.RED
        status("Predicted",
               f"{prediction_result.apogee_m:.1f} m  ({prediction_result.apogee_ft:.0f} ft)", col)
        status("Error",    f"{err_m:+.1f} m  ({err_pc:+.2f}%)", col)
        status("Latency",  f"{prediction_result.elapsed_ms:.1f} ms")
    else:
        status("Prediction", "Not completed")

    print()


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rocket Apogee Prediction Demo")
    p.add_argument("--model",  choices=["mlp", "rf", "lr"], default="mlp",
                   help="ML model to use (default: mlp)")
    p.add_argument("--flight", type=int, default=0,
                   help="Flight index in dataset (default: 0)")
    p.add_argument("--no-audio", action="store_true",
                   help="Disable audio playback")
    p.add_argument("--serial", metavar="PORT", default=None,
                   help="Arduino serial port, e.g. /dev/ttyUSB0")
    p.add_argument("--baud",   type=int, default=115200)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    run_demo(
        model_type   = args.model,
        flight_index = args.flight,
        no_audio     = args.no_audio,
        serial_port  = args.serial,
        baud         = args.baud,
    )
