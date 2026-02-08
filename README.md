# Rocket Apogee Prediction

This project trains machine learning models to predict rocket apogee from flight telemetry data. It uses a sliding window approach to analyze recent flight data and predict the final apogee.

## Quick Start

### 1. Environment Setup

Create and activate a virtual environment, then install dependencies:

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Workflow

The project follows a numbered script workflow for data generation, processing, training, and analysis.

**1. Data Generation**
Generate a dataset of simulated flights using RocketPy. This creates `data/raw/batch_dataset_v1.csv`.
```bash
python scripts/1_batch_simulation_creation.py
```

**2. Data Processing**
Process the raw simulation data into sliding windows suitable for training. This creates training and testing CSVs in `data/processed/`.
```bash
python scripts/2_sliding_window_generator.py
```

**3. Model Training**
Train the machine learning models (MLP, Random Forest, Linear Regression). Models are saved to `models/`.
```bash
python scripts/3_model_creation.py
```

**4. Testing**
Evaluate the trained models on specific flights or the entire test set.
```bash
python scripts/4_model_testing.py --model mlp
# Options: --model [mlp|random_forest|linear_regression]
# Use --help for more options
```

**5. Analysis**
Perform detailed analysis of model performance, including overfitting checks and error distribution over time.
```bash
python scripts/5_model_analysis.py
```

**6. Visualization**
Generate comprehensive visualization plots comparing all models. plots are saved to `visualizations/`.
```bash
python scripts/6_visualization_suite.py
```

### 3. GUI Application

A Tkinter-based GUI is provided to run these steps and visualize predictions interactively.

```bash
python gui.py
```
*   **Generate Data**: options to run the Simulation and Sliding Window scripts.
*   **Train**: Button to retrain models.
*   **Predict**: Run the "Apogee Tests" to visualize predictions for a specific flight or average performance.

## Project Structure

```
.
├── data/
│   ├── raw/                              # Original simulated flight CSVs
│   ├── processed/                        # Sliding window training/test sets
│   └── scalers/                          # Saved scaler files for data normalization
├── models/
│   ├── apogee_mlp_model.pth              # PyTorch MLP model
│   ├── apogee_random_forest.pkl          # Scikit-learn Random Forest
│   └── apogee_linear_regression.pkl      # Scikit-learn Linear Regression
├── scripts/
│   ├── 1_batch_simulation_creation.py    # Generate flight data with RocketPy
│   ├── 2_sliding_window_generator.py     # Create sliding windows from data
│   ├── 3_model_creation.py               # Train and save models
│   ├── 4_model_testing.py                # Evaluate single models
│   ├── 5_model_analysis.py               # Deep dive performance analysis
│   └── 6_visualization_suite.py          # Generate comparison plots
├── gui.py                                # Main GUI application
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

## Model Info

*   **Input**: 2.5s window of telemetry (Altitude, Velocity, Acceleration, etc.)
*   **Target**: Final Apogee Altitude
*   **Models**:
    *   **MLP**: 3-layer Neural Network
    *   **Random Forest**: Ensemble regressor
    *   **Linear Regression**: Baseline linear model
