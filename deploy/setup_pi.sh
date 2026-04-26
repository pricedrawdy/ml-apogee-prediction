#!/usr/bin/env bash
# setup_pi.sh — One-shot setup for RocketPy Apogee Prediction on Raspberry Pi
# =============================================================================
# Tested on:  Raspberry Pi OS Bookworm / Bullseye  (64-bit ARM)
# Run once:   chmod +x deploy/setup_pi.sh && ./deploy/setup_pi.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_DIR="$REPO_ROOT/deploy"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   RocketPy — Raspberry Pi Apogee Prediction Setup        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Repo root : $REPO_ROOT"
echo ""

# ── 1. System packages ────────────────────────────────────────────────────────
echo "▸ Installing system packages …"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-pip \
    python3-venv \
    python3-numpy \
    libatlas-base-dev \
    libopenblas-dev \
    libasound2-dev \
    libsdl2-mixer-2.0-0 \
    portaudio19-dev \
    git

# ── 2. Python virtual environment ─────────────────────────────────────────────
VENV="$REPO_ROOT/venv_pi"
if [ ! -d "$VENV" ]; then
    echo "▸ Creating Python virtual environment at $VENV …"
    python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"

echo "▸ Upgrading pip …"
pip install --upgrade pip wheel --quiet

# ── 3. Python dependencies ────────────────────────────────────────────────────
echo "▸ Installing Python packages (this may take a few minutes on Pi 4/5) …"
pip install --quiet -r "$DEPLOY_DIR/requirements_pi.txt"

# ── 4. Generate sound assets ──────────────────────────────────────────────────
echo "▸ Generating sound files …"
python "$DEPLOY_DIR/download_sounds.py"

# ── 5. Verify models exist ────────────────────────────────────────────────────
echo ""
echo "▸ Checking for trained model files …"
MODELS_OK=true
for f in \
    "$REPO_ROOT/models/apogee_mlp_model.pth" \
    "$REPO_ROOT/models/apogee_random_forest.pkl" \
    "$REPO_ROOT/models/apogee_linear_regression.pkl" \
    "$REPO_ROOT/data/scalers/apogee_input_scaler.pkl" \
    "$REPO_ROOT/data/scalers/apogee_target_scaler.pkl"
do
    if [ -f "$f" ]; then
        echo "  ✅ $(basename $f)"
    else
        echo "  ❌ MISSING: $f"
        MODELS_OK=false
    fi
done

if [ "$MODELS_OK" = false ]; then
    echo ""
    echo "⚠️  One or more model files are missing."
    echo "   Copy the models/ and data/scalers/ directories from your"
    echo "   development machine to the Pi before running the demo."
    echo "   Example (run on dev machine):"
    echo "     rsync -avz models/ data/scalers/ pi@raspberrypi.local:~/RocketPy/models/"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   Setup complete!                                         ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Activate venv:  source $VENV/bin/activate               "
echo "║  Run demo:       python deploy/flight_demo.py            "
echo "║  Run headless:   python deploy/flight_demo.py --no-audio "
echo "║  With Arduino:   python deploy/flight_demo.py \\          "
echo "║                      --serial /dev/ttyUSB0              "
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
