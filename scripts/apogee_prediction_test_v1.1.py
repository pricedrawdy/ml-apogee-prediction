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
flight_index = 2  # CHANGE THIS to test a different flight (0-based)

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



# %%
# Update the plotting section
import matplotlib.pyplot as plt

# Correct file path using Path
data_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
df_full = pd.read_csv(data_dir / "batch_dataset_v1.csv")

# Choose the Nth row (flight) from the original unwindowed data
flight_index = 0  # change this if needed

# Compute sliding window parameters
timestep_interval = 0.025
window_size = int(2.5 / timestep_interval)
stride = int(0.25 / timestep_interval)

# Calculate samples per flight
samples_per_flight = ((25 / timestep_interval - window_size) // stride) + 1
samples_per_flight = int(samples_per_flight)

# Get the window range for that one flight
start = flight_index * samples_per_flight
end = start + samples_per_flight

# Slice predictions and true values
preds = y_pred[start:end].flatten()
true_vals = y_true[start:end].flatten()

# Calculate time points for x-axis
time = np.array([((i * stride) + window_size // 2) * timestep_interval 
                 for i in range(len(preds))])

# Limit the prediction window to first 10 seconds
max_flight_time = 10.0  # seconds
time_mask = time <= max_flight_time

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(time[time_mask], preds[time_mask], 'b-o', 
         label='Predicted Apogee', markersize=4)
plt.hlines(target_apogee, time[0], max_flight_time, 
          colors='r', linestyles='--', label='True Apogee')

# Add error bands (±5% of true apogee)
error_margin = 0.05 * target_apogee
plt.fill_between(time[time_mask], 
                target_apogee - error_margin, 
                target_apogee + error_margin, 
                color='r', alpha=0.1)

plt.title("Apogee Prediction Over Time (Sliding Window)")
plt.xlabel("Time into Flight (s)")
plt.ylabel("Apogee Prediction (m)")
plt.xlim(0, max_flight_time)  # Set x-axis limits from 0 to max_flight_time
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Add error statistics
error = np.abs(preds - target_apogee)
print(f"\nError Statistics for Flight {flight_index}:")
print(f"Mean Absolute Error: {np.mean(error):.2f} m")
print(f"Max Error: {np.max(error):.2f} m")
print(f"Min Error: {np.min(error):.2f} m")
