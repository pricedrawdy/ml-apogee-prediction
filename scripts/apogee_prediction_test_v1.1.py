# %%
from pathlib import Path
import torch
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# %%
# === Load trained model and scalers ===
scalers_dir = Path(__file__).resolve().parent.parent / "data" / "scalers" 
model_dir = Path(__file__).resolve().parent.parent / "models"

input_scaler = joblib.load(scalers_dir / "apogee_input_scaler.pkl")
target_scaler = joblib.load(scalers_dir / "apogee_target_scaler.pkl")

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

# Load model weights
model = ApogeeMLP(input_dim=400)  # adjust if your input shape differs
model.load_state_dict(torch.load(model_dir / "apogee_mlp_model.pth"))
model.eval()

# %%
# === Load the dataset ===
input_csv = "sliding_test_by_flight.csv"  # The test dataset file
input_csv_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
df = pd.read_csv(input_csv_dir / input_csv)

X = df.drop(columns=["Apogee"]).values
y_true = df["Apogee"].values.reshape(-1, 1)

# Scale inputs
X_scaled = input_scaler.transform(X)

# %%
# Predict with the model
with torch.no_grad():
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_pred_scaled = model(X_tensor).numpy()

# Unscale predicted apogees
y_pred = target_scaler.inverse_transform(y_pred_scaled)

# === Evaluation ===
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.2f} meters")

# === Settings ===
flight_index = 2  # 👈 CHANGE THIS to test a different flight (0-based)

# === Sliding window parameters ===
timestep_interval = 0.025
window_size = int(2.5 / timestep_interval)  # e.g., 100 steps
stride = int(0.25 / timestep_interval)      # e.g., 10 steps

# === Group samples by unique apogee ===
unique_apogees = np.unique(y_true)

# Choose which flight to display
flight_index = 2  # Change this to view another flight
target_apogee = unique_apogees[flight_index]

# Find all rows for that apogee
indices = np.where(np.isclose(y_true, target_apogee, atol=0.01))[0]

print(f"\n=== Predictions for Flight {flight_index} with True Apogee ~= {target_apogee:.2f} m ===")
for i in indices:
    print(f"Sample {i}: Pred = {y_pred[i][0]:.2f} m, True = {y_true[i][0]:.2f} m")

# Summary stats for the selected flight
preds = y_pred[indices].flatten()
true_val = float(target_apogee)
errors = preds - true_val

print("\n--- Flight summary ---")
print(f"windows={len(preds)}  mean_pred={preds.mean():.2f}  median_pred={np.median(preds):.2f}")
print(f"bias(mean error)={errors.mean():.2f} m  MAE={np.mean(np.abs(errors)):.2f} m  RMSE={np.sqrt(np.mean(errors**2)):.2f} m")
print(f"min_pred={preds.min():.2f}  max_pred={preds.max():.2f}  std={preds.std(ddof=1):.2f} m")


# %%
import matplotlib.pyplot as plt

# Load unwindowed dataset with corrected path
processed_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
df_full = pd.read_csv(processed_dir / "batch_dataset_v1.csv")

# Choose the Nth row (flight) from the original unwindowed data
flight_index = 0  # change this if needed

# Compute sliding window parameters
timestep_interval = 0.025
window_size = int(2.5 / timestep_interval)  # 100
stride = int(0.25 / timestep_interval)      # 10

# Calculate how many windows per flight
# Total steps per flight = 25s / 0.025s = 1000
total_steps = int(25 / timestep_interval)
samples_per_flight = ((total_steps - window_size) // stride) + 1  # Should be 91

# Get the window range for that one flight
start = flight_index * samples_per_flight
end = start + samples_per_flight

# Slice predictions and true values
preds = y_pred[start:end].flatten()
true_vals = y_true[start:end].flatten()

# Get target apogee from df_full if it has an Apogee column, otherwise use first true value
if "Apogee" in df_full.columns:
    target_apogee = df_full.iloc[flight_index]["Apogee"]
else:
    target_apogee = true_vals[0]

# Time is at the center of each window
time = np.array([((i * stride) + window_size // 2) * timestep_interval for i in range(len(preds))])

# Cap at 10 seconds
mask = time <= 10.0
time = time[mask]
preds = preds[mask]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(time, preds, label='Predicted Apogee', marker='o')
plt.hlines(target_apogee, time[0], time[-1], colors='r', linestyles='--', label='True Apogee')

plt.title(f"Apogee Prediction Over Time - Flight {flight_index} (Sliding Window, ≤10s)")
plt.xlabel("Time into Flight (s)")
plt.ylabel("Apogee Prediction (m)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
