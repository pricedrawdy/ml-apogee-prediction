===============================
RocketSerializer Quick Guide
===============================

To convert an OpenRocket .ork file to JSON using RocketSerializer:

-------------
STEP 1: Open Terminal
-------------
Open Command Prompt (or PowerShell) and navigate to your RocketPy folder:

    cd C:\Users\price\Documents\RocketPy

-------------
STEP 2: Activate Virtual Environment
-------------
Activate the Python virtual environment:

    .\venv\Scripts\activate

-------------
STEP 3: Set JAVA_HOME (only needed if not already set globally)
-------------
If JAVA_HOME is not set permanently on your system, run:

    set JAVA_HOME=C:\Program Files\Java\jdk-17
    set PATH=%JAVA_HOME%\bin;%PATH%

-------------
STEP 4: Run RocketSerializer
-------------
Convert your .ork file to JSON using the following command:

    ork2json --filepath Trinity9.4.ork --ork_jar "C:/Program Files/OpenRocket/OpenRocket.jar" --output ./json_output

This will generate a JSON version of your `.ork` file and place it inside the `json_output` folder (which will be created if it doesn't exist).

-------------
NOTES:
-------------
- Make sure `OpenRocket.jar` exists at the specified path.
- You can replace `Trinity9.4.ork` with any other `.ork` file in the folder.
- You must be connected to the internet the first time you run RocketSerializer (for any dependency downloads).

===============================


===============================
# 🚀 Rocket Apogee Prediction
===============================

This repository contains a machine learning pipeline for predicting rocket apogee using real-time flight telemetry and a custom sliding window generator. The model is built using PyTorch and trained on a dataset of ~120 real flights.

## 📁 Project Structure

```

.
apogee-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                     # Original CSV with all 120+ flights
│   ├── processed/               # Train/test CSVs created by sliding window generator
│   └── scalers/                 # Scaler .pkl files
├── models/
│   ├── apogee_mlp_model.pth     # Saved model weights
│   └── architecture.py          # MLP class (ApogeeMLP)
├── scripts/
│   ├── sliding_window_generator.py
│   ├── train_model.py
│   ├── evaluate_apogee.py       # Your test script (refactored for multi-flight)
│   └── helpers.py               # Shared functions (e.g., load_scalers, plot_preds)
├── notebooks/
│   └── analysis.ipynb           # For interactive EDA, RMSE vs time, etc.
└── deploy/
    ├── onboard_pipeline.py      # Minimal real-time prediction for flight computer
    └── config.yaml              # Configs (e.g., scaler paths, window size)


````

---

## 🔧 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/pricedrawdy/ml-apogee-prediction.git
cd ml-apogee-prediction
````

### 2. Create a Virtual Environment

**Windows:**

```cmd
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Generate Sliding Windows

```bash
python scripts/sliding_window_generator.py
```

### Train the Model

```bash
python scripts/train_mlp_model.py
```

### Evaluate Predictions (on Test Set)

```bash
python scripts/evaluate_prediction.py
```

---

## 📌 Notes

* All file paths use Python’s `pathlib` for cross-platform compatibility.
* Model weights and scalers are saved using `torch.save()` and `joblib`.
* Data must be in the `data/` directory to work with the provided scripts.

---

## 🧠 Model Overview

* Input: Flattened sliding windows of time-series flight data (e.g., velocity, altitude, etc.)
* Output: Predicted apogee (in meters)
* Architecture: 3-layer MLP (128 → 64 → 1)

---

## 🤝 Contributing

This is a private academic research project. Contributions welcome if you’re on the dev team.

---

## 🧪 Environment Info

* Python 3.10+
* PyTorch ≥ 2.0
* scikit-learn ≥ 1.0
* pandas, numpy

===============================
