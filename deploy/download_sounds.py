"""
download_sounds.py — Fetch / generate audio files for the flight demo.

Generates WAV files programmatically using numpy (no network needed).
Place in deploy/ and run:

    python download_sounds.py

Generates:
    sounds/igniter_arm.wav     — electronic beep sequence (T-10 countdown)
    sounds/ignition.wav        — short sharp bang (ignition)
    sounds/motor_burn.wav      — 8 s motor roar (loops during burn)
    sounds/burnout.wav         — rumble cut to silence
    sounds/prediction_beep.wav — ascending confirmation tones
    sounds/apogee_beep.wav     — long high beep at apogee
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

SOUNDS = Path(__file__).resolve().parent / "sounds"
SOUNDS.mkdir(exist_ok=True)

SR = 44100   # samples / second


def _write_wav(path: Path, samples, sr: int = SR) -> None:
    """Write normalised float samples to a 16-bit WAV file."""
    import numpy as np
    data = np.array(samples, dtype=np.float32)
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak * 0.90
    pcm = (data * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    print(f"  ✅ {path.name}")


def _beep(freq: float, duration: float, sr: int = SR, amp: float = 0.8) -> list:
    import numpy as np
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    env = np.ones_like(t)
    # Soft attack / decay
    attack = int(0.01 * sr)
    decay  = int(0.03 * sr)
    env[:attack]  = np.linspace(0, 1, attack)
    env[-decay:]  = np.linspace(1, 0, decay)
    return (amp * env * np.sin(2 * math.pi * freq * t)).tolist()


def _noise(duration: float, colour: str = "white", sr: int = SR) -> "np.ndarray":
    import numpy as np
    n = int(sr * duration)
    w = np.random.randn(n)
    if colour == "pink":
        # Crude pink noise via IIR
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        from scipy.signal import lfilter
        try:
            w = lfilter(b, a, w)
        except ImportError:
            pass  # fall back to white
    return w


def gen_igniter_arm(sr: int = SR) -> None:
    import numpy as np
    samples = []
    for freq in [880, 1100, 1320]:
        samples += _beep(freq, 0.12, sr)
        samples += [0.0] * int(sr * 0.05)
    _write_wav(SOUNDS / "igniter_arm.wav", samples)


def gen_ignition(sr: int = SR) -> None:
    import numpy as np
    dur = 0.8
    t = np.linspace(0, dur, int(sr * dur))
    # Crackle: band-pass noise
    noise = _noise(dur)
    # Sharp transient at start
    env = np.exp(-t * 8) + 0.15 * np.exp(-t * 2)
    sig = noise * env * 0.9
    _write_wav(SOUNDS / "ignition.wav", sig)


def gen_motor_burn(sr: int = SR) -> None:
    import numpy as np
    dur = 8.0  # loop-able
    t = np.linspace(0, dur, int(sr * dur))
    noise = _noise(dur) * 0.5
    # Low-frequency rumble
    rumble = 0.3 * np.sin(2 * math.pi * 60  * t) \
           + 0.2 * np.sin(2 * math.pi * 120 * t) \
           + 0.15* np.sin(2 * math.pi * 240 * t)
    # Slight swell
    env = 0.85 + 0.15 * np.sin(2 * math.pi * 0.5 * t)
    sig = (noise + rumble) * env
    _write_wav(SOUNDS / "motor_burn.wav", sig)


def gen_burnout(sr: int = SR) -> None:
    import numpy as np
    dur = 1.2
    t = np.linspace(0, dur, int(sr * dur))
    noise = _noise(dur) * np.exp(-t * 4)
    _write_wav(SOUNDS / "burnout.wav", noise * 0.7)


def gen_prediction_beep(sr: int = SR) -> None:
    samples = []
    for freq in [660, 880, 1100, 1320]:
        samples += _beep(freq, 0.10, sr, amp=0.6)
        samples += [0.0] * int(sr * 0.03)
    _write_wav(SOUNDS / "prediction_beep.wav", samples)


def gen_apogee_beep(sr: int = SR) -> None:
    samples = _beep(1760, 1.2, sr, amp=0.75)
    _write_wav(SOUNDS / "apogee_beep.wav", samples)


def main() -> None:
    print("Generating sound assets …")
    gen_igniter_arm()
    gen_ignition()
    gen_motor_burn()
    gen_burnout()
    gen_prediction_beep()
    gen_apogee_beep()
    print("\nAll sounds generated in:", SOUNDS)


if __name__ == "__main__":
    main()
