# Rocket Apogee Prediction

This project trains a neural network to predict rocket apogee from flight telemetry. Sliding windows of recent data are used as input to a PyTorch model.

## 1. Create a rocket in OpenRocket

## 2. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv         # on Mac use python3 -m venv venv
venv\Scripts\activate       # on Mac use source venv/bin/activate
pip install -r requirements.txt
```
## 3. Use RocketSerializer to convert your OpenRocket into use with RocketPy
- Note: on Mac getting the 

Find your OpenRocket .ork file and OpenRocket .jar file
- On Mac the .jar can be found under /Applications/OpenRocket.app/Contents/Resources/app/jar
- Note: in order to get this to run on mac you will have to mess around with your Java installation. Windows is simpler
- On windows, find the jar yourself
- Put your .ork file in rocket-info
- Swap the file names/paths with your file names/paths
```bash
ork2json --filepath <file.ork> --ork_jar <path/to/OpenRocket.jar> --output ./json_output
```

## 4. Place converted files in rocket-info

## 5. Copy rocket data into batch_simulation_creation from parameters.json

## 6. Generate batch dataset (can take 20+ minutes depending on the parameter variation)

```bash
python scripts/batch_simulation_creation.py
```

## 7. Generate training windows:

```bash
python scripts/sliding_window_generator_v2.py
```

## 8. Train the model:

```bash
python models/apogee_prediction_model_v1.py
```

## 9. Evaluate on the test set:

```bash
python scripts/apogee_prediction_test_v1.1.py
```

## Project Structure

```
.
├── data/
│   ├── raw/                              # Original flight CSVs
│   ├── processed/                        # Train/test windows
│   └── scalers/                          # Saved scaler files
├── models/
│   └── apogee_prediction_model_v1.py     # Training script
├── scripts/
│   ├── apogee_prediction_test_v1.1.py   # Evaluate predictions
│   ├── batch_simulation_creation.py     # Create simulation data
│   └── sliding_window_generator_v2.py   # Generate windows
├── notebooks/                           # Example Jupyter notebooks for testing
└── deploy/                              # Real-time prediction pipeline
```

## Model Overview

- Input: 2.5 s window of telemetry
- Architecture: 3-layer MLP
- Output: Predicted apogee (meters)


---

## Contributing

This repository is used for academic research. Contributions are welcome via pull requests.

## Environment

- Python 3.10+
- PyTorch 2.0+
- scikit-learn 1.0+
