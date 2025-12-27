from pathlib import Path
import torch
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# === Configuration ===
FLIGHT_INDEX = 2            # Flight to analyze/plot (0-based)
TIMESTEP = 0.025            # Seconds
WINDOW_DURATION = 2.5       # Seconds
STRIDE_DURATION = 0.25      # Seconds
TOTAL_FLIGHT_TIME = 25.0    # Seconds (Total duration per flight in dataset)
PLOT_MAX_TIME = 10.0        # Seconds (Limit x-axis for visibility)

# Calculated parameters
WINDOW_SIZE = int(WINDOW_DURATION / TIMESTEP)
STRIDE = int(STRIDE_DURATION / TIMESTEP)
SAMPLES_PER_FLIGHT = int(((TOTAL_FLIGHT_TIME / TIMESTEP - WINDOW_SIZE) // STRIDE) + 1)

# === Model Definition ===
class ApogeeMLP(torch.nn.Module):
    def __init__(self, input_dim):
        super(ApogeeMLP, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

# === Setup Paths ===
base_dir = Path(__file__).resolve().parent.parent
scalers_dir = base_dir / "data" / "scalers"
model_dir = base_dir / "models"
data_dir = base_dir / "data" / "processed"

# === Load Resources ===
print("Loading model and scalers...")
input_scaler = joblib.load(scalers_dir / "apogee_input_scaler.pkl")
target_scaler = joblib.load(scalers_dir / "apogee_target_scaler.pkl")

# Load Data
df = pd.read_csv(data_dir / "sliding_test_by_flight.csv")
X = df.drop(columns=["Apogee"]).values
y_true = df["Apogee"].values.reshape(-1, 1)

# Load Model
model = ApogeeMLP(input_dim=X.shape[1])
model.load_state_dict(torch.load(model_dir / "apogee_mlp_model.pth"))
model.eval()

# === Inference ===
print("Running inference...")
X_scaled = input_scaler.transform(X)

with torch.no_grad():
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_pred_scaled = model(X_tensor).numpy()

y_pred = target_scaler.inverse_transform(y_pred_scaled)

# === Global Evaluation ===
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
print(f"Global RMSE: {rmse:.2f} meters")

# === Single Flight Analysis ===
# Calculate indices for the specific flight
start_idx = FLIGHT_INDEX * SAMPLES_PER_FLIGHT
end_idx = start_idx + SAMPLES_PER_FLIGHT

if start_idx >= len(y_pred):
    raise ValueError(f"Flight index {FLIGHT_INDEX} out of bounds.")

# Slice data
flight_preds = y_pred[start_idx:end_idx].flatten()
flight_true = y_true[start_idx:end_idx].flatten()
target_apogee = flight_true[0] # Assuming constant apogee per flight

# Generate time axis
time_axis = np.array([((i * STRIDE) + WINDOW_SIZE // 2) * TIMESTEP 
                      for i in range(len(flight_preds))])

# Calculate Error Stats
error = np.abs(flight_preds - target_apogee)
print(f"\n=== Analysis for Flight {FLIGHT_INDEX} ===")
print(f"True Apogee: {target_apogee:.2f} m")
print(f"Mean Absolute Error: {np.mean(error):.2f} m")
print(f"Max Error: {np.max(error):.2f} m")

# === Plotting ===
# Filter for max plot time
mask = time_axis <= PLOT_MAX_TIME

plt.figure(figsize=(12, 6))
plt.plot(time_axis[mask], flight_preds[mask], 'b-o', label='Predicted Apogee', markersize=4)
plt.hlines(target_apogee, time_axis[0], PLOT_MAX_TIME, colors='r', linestyles='--', label='True Apogee')

# Error bands (±1%)
error_margin = 0.01 * target_apogee
plt.fill_between(time_axis[mask], 
                 target_apogee - error_margin, 
                 target_apogee + error_margin, 
                 color='r', alpha=0.1, label='±1% Margin')

plt.title(f"Apogee Prediction: Flight {FLIGHT_INDEX}")
plt.xlabel("Time into Flight (s)")
plt.ylabel("Apogee (m)")
plt.xlim(0, PLOT_MAX_TIME)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()