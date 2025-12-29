# %%
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# %%
# === CONFIGURATION ===
input_csv = "batch_dataset_v1.csv"  # The original dataset file
window_seconds = 2.5  # Length of input buffer (in seconds)
step_seconds = 0.25   # Slide forward every N seconds
timestep_interval = 0.025  # How far apart each original timestep is (s)

# === LOAD DATA ===
input_csv_dir = Path(__file__).resolve().parent.parent / "data" / "raw"    # The original dataset
df_full = pd.read_csv(input_csv_dir / input_csv)

# Fill NaNs with 0 (post-apogee padding)
df_full.fillna(0, inplace=True)

# Count initial number of columns
initial_cols = df_full.shape[1]

# Drop all-zero columns
df = df_full.loc[:, (df_full != 0).any(axis=0)]

# Count and report how many were removed
removed_cols = initial_cols - df.shape[1]
print(f"Removed {removed_cols} all-zero columns.")

# %%
# Compute sliding window size and stride
window_size = int(window_seconds / timestep_interval)
stride = int(step_seconds / timestep_interval)

# Define feature groups
features_per_timestep = ["Vertical velocity", "Vertical acceleration", "Total velocity", "Altitude"]
feature_groups = {
    label: sorted([col for col in df.columns if col.startswith(label)])
    for label in features_per_timestep
}

# Keep static flight descriptors alongside each window (e.g., wind, temperature, launch angle)
static_columns = [
    col
    for col in df.columns
    if not any(col.startswith(prefix) for prefix in features_per_timestep)
    and col not in ["Apogee altitude (m)", "Apogee time (s)"]
]

# Split entire flights (rows) into train/test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# %%
# Sliding window function
def generate_windows(df):
    samples = []
    targets = []
    for _, row in df.iterrows():
        static_vals = row[static_columns].to_numpy()
        # Create a 2D array: shape (num_features, num_timesteps)
        series = np.vstack([
            row[feature_groups["Vertical velocity"]],
            row[feature_groups["Vertical acceleration"]],
            row[feature_groups["Total velocity"]],
            row[feature_groups["Altitude"]],
        ])
        apogee = row["Apogee altitude (m)"]
        max_start = series.shape[1] - window_size

        for start in range(0, max_start + 1, stride):
            window = series[:, start:start + window_size]
            if window.shape[1] == window_size:
                window_features = np.concatenate([static_vals, window.flatten()])
                samples.append(window_features)
                targets.append(apogee)

    return pd.DataFrame(samples), pd.Series(targets)


# %%
# Generate sliding windows
X_train, y_train = generate_windows(train_df)
X_test, y_test = generate_windows(test_df)

# Combine into labeled datasets
train_set = X_train.copy()
train_set["Apogee"] = y_train
test_set = X_test.copy()
test_set["Apogee"] = y_test

# %%
# Save for future use
model_dir = Path(__file__).resolve().parent.parent / "data" / "processed" 
model_dir.mkdir(parents=True, exist_ok=True)

train_model_path = model_dir / "sliding_train_by_flight.csv"
test_model_path = model_dir / "sliding_test_by_flight.csv"

train_set.to_csv(train_model_path, index=False)
test_set.to_csv(test_model_path, index=False)

print("Sliding window datasets saved:", train_model_path, test_model_path)
