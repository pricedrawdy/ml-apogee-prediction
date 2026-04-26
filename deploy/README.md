# Raspberry Pi Deployment — Apogee Prediction Demo

## Overview

This package runs a **fake flight scenario** on a Raspberry Pi (5 / 4 / Zero 2W).
It does **not** require the M.2 AI HAT+; inference runs fine on the Pi CPU using scikit-learn / PyTorch.

### What happens

1. Countdown — 10 s idle before launch
2. **Liftoff** — realistic audio + live telemetry stream begins
3. **Burn phase** — motor burn ~4.7 s, noisy sensor values printed in real time
4. **Burnout detected** — ML model runs on the 2.5 s sliding window and prints the apogee prediction
5. **Coast / apogee** — telemetry continues until apogee, true value revealed
6. Summary shown (predicted vs actual, error %)

---

## Files

| File | Purpose |
|------|---------|
| `flight_demo.py` | Main demo program |
| `fake_flight_generator.py` | Synthesises a realistic faux flight profile |
| `inference_engine.py` | Wraps MLP / RF / LR models for single-window inference |
| `sounds/` | WAV audio files (generated / downloaded at setup) |
| `setup_pi.sh` | One-shot setup script for a fresh Pi OS install |
| `requirements_pi.txt` | Minimal dependency list for the Pi |

---

## Quick Start

```bash
# On the Pi (Raspberry Pi OS Bookworm / Bullseye 64-bit)
cd ~/RocketPy/deploy
chmod +x setup_pi.sh
./setup_pi.sh          # installs deps + downloads sounds

python flight_demo.py              # default: MLP model
python flight_demo.py --model rf   # Random Forest
python flight_demo.py --model lr   # Linear Regression
python flight_demo.py --no-audio   # silent mode (SSH / headless)
```

---

## Optional: Arduino serial feed

If you want real faux sensor data streamed from an Arduino over USB:

```bash
python flight_demo.py --serial /dev/ttyUSB0 --baud 115200
```

Upload `arduino/fake_flight_sensor.ino` to the Arduino.
The Pi will read NMEA-style CSV lines and feed them into the same inference pipeline.
