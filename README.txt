# 🚀 Rocket Apogee Prediction

This project trains a neural network to predict rocket apogee from flight telemetry. Sliding windows of recent data are used as input to a PyTorch model.

## Quick Start

1. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
source venv/bin/activate    # on Windows use venv\Scripts\activate
pip install -r requirements.txt
```

2. Place your raw CSVs in `data/raw/`.

3. Generate training windows:

```bash
python scripts/sliding_window_generator_v2.py
```

4. Train the model:

```bash
python models/apogee_prediction_model_v1.py
```

5. Evaluate on the test set:

```bash
python scripts/apogee_prediction_test_v1.1.py
```

## Project Structure

```
.
├── data/
│   ├── raw/                  # Original flight CSVs
│   ├── processed/            # Train/test windows
│   └── scalers/              # Saved scaler files
├── models/
│   └── apogee_prediction_model_v1.py  # Training script
├── scripts/
│   ├── apogee_prediction_test_v1.1.py # Evaluate predictions
│   ├── batch_simulation_creation.py   # Create simulation data
│   └── sliding_window_generator_v2.py # Generate windows
├── notebooks/               # Example Jupyter notebooks
└── deploy/                  # Real-time prediction pipeline
```

## Model Overview

- Input: 2.5 s window of telemetry
- Architecture: 3-layer MLP
- Output: Predicted apogee (meters)

---

### RocketSerializer (optional)

If you need JSON versions of OpenRocket `.ork` files, install RocketSerializer and run:

```bash
ork2json --filepath <file.ork> --ork_jar <path/to/OpenRocket.jar> --output ./json_output
```

---

## Contributing

This repository is used for academic research. Contributions are welcome via pull requests.

## Environment

- Python 3.10+
- PyTorch 2.0+
- scikit-learn 1.0+
