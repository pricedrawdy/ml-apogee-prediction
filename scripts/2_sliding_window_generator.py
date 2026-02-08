# %%
"""
Sliding Window Generator v2 - Burnout Phase Prediction

This version prevents data leakage by:
1. Only using windows from the burnout phase (5-8 seconds)
2. Using relative altitude (change within window) instead of raw altitude
3. All features are real-time measurable from flight computer sensors

Features per window:
- Static: wind speed, temperature, launch angle
- Time-series: vertical velocity, vertical acceleration, total velocity,
              horizontal velocity, pitch angle, dynamic pressure, Mach number, pressure
- Derived: relative altitude (delta within window), velocity squared (energy proxy)
"""
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

# Burnout phase timing - only use windows in this range
# Burnout occurs around 4.7s, so we capture windows ending 5-8s into flight
BURNOUT_WINDOW_START = 5.0  # Earliest window END time (seconds)
BURNOUT_WINDOW_END = 8.0    # Latest window END time (seconds)

# === LOAD DATA ===
input_csv_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
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

# Calculate valid window start indices for burnout phase
# Window end time = (start_index + window_size) * timestep_interval
# So start_index for end_time T is: (T / timestep_interval) - window_size
min_start_idx = int(BURNOUT_WINDOW_START / timestep_interval) - window_size
max_start_idx = int(BURNOUT_WINDOW_END / timestep_interval) - window_size
print(f"Burnout phase: window start indices {min_start_idx} to {max_start_idx}")

# Define feature groups - NOTE: We still load Altitude but use it differently
raw_features = [
    "Vertical velocity", "Vertical acceleration", "Total velocity",
    "Horizontal velocity", "Pitch angle", "Dynamic pressure", 
    "Mach number", "Pressure"
]
altitude_label = "Altitude"

feature_groups = {
    label: sorted([col for col in df.columns if col.startswith(label)])
    for label in raw_features + [altitude_label]
}

# NOTE: We intentionally EXCLUDE static features (wind, temp, launch angle)
# because in the simulation, apogee is deterministically determined by launch angle.
# Including them would allow the model to trivially predict apogee without 
# learning from telemetry patterns. In real deployment, these values might not
# be known accurately anyway.
static_columns = []  # Empty - we only use dynamic telemetry features

# Split entire flights (rows) into train/test BEFORE generating windows
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Train flights: {len(train_df)}, Test flights: {len(test_df)}")

# %%
def generate_burnout_windows(df):
    """
    Generate sliding windows only from the burnout phase.
    
    Features (all real-time measurable):
    - Static: wind, temp, launch angle
    - Vertical velocity time series
    - Vertical acceleration time series  
    - Total velocity time series
    - Horizontal velocity time series
    - Pitch angle time series
    - Dynamic pressure time series
    - Mach number time series
    - Pressure time series
    - Relative altitude (altitude change within window) - NOT raw altitude
    - Velocity squared (kinetic energy proxy)
    """
    samples = []
    targets = []
    
    for _, row in df.iterrows():
        static_vals = row[static_columns].to_numpy()
        
        # Load raw time series
        v_vel = np.array(row[feature_groups["Vertical velocity"]])
        v_acc = np.array(row[feature_groups["Vertical acceleration"]])
        t_vel = np.array(row[feature_groups["Total velocity"]])
        altitude = np.array(row[feature_groups["Altitude"]])
        h_vel = np.array(row[feature_groups["Horizontal velocity"]])
        pitch = np.array(row[feature_groups["Pitch angle"]])
        dynp = np.array(row[feature_groups["Dynamic pressure"]])
        mach = np.array(row[feature_groups["Mach number"]])
        pressure = np.array(row[feature_groups["Pressure"]])
        
        apogee = row["Apogee altitude (m)"]
        max_possible_start = len(v_vel) - window_size
        
        # Only generate windows in the burnout phase
        for start in range(max(0, min_start_idx), min(max_possible_start + 1, max_start_idx + 1), stride):
            end = start + window_size
            
            # Extract window data
            window_v_vel = v_vel[start:end]
            window_v_acc = v_acc[start:end]
            window_t_vel = t_vel[start:end]
            window_alt = altitude[start:end]
            window_h_vel = h_vel[start:end]
            window_pitch = pitch[start:end]
            window_dynp = dynp[start:end]
            window_mach = mach[start:end]
            window_pressure = pressure[start:end]
            
            # Skip if window contains NaN/zero padding (past apogee)
            if np.any(window_v_vel == 0) and np.any(window_alt == 0):
                continue
            
            # === DERIVED FEATURES (real-time computable) ===
            # Relative altitude: change from start of window (prevents leakage)
            relative_alt = window_alt - window_alt[0]
            
            # Velocity squared as kinetic energy proxy
            vel_squared = window_t_vel ** 2
            
            # Combine all features
            window_features = np.concatenate([
                static_vals,
                window_v_vel,        # Vertical velocity (from accelerometer + integration)
                window_v_acc,        # Vertical acceleration (from accelerometer)
                window_t_vel,        # Total velocity (from accelerometer)
                window_h_vel,        # Horizontal velocity (from GPS/IMU)
                window_pitch,        # Pitch angle (from IMU)
                window_dynp,         # Dynamic pressure (from pitot/derived)
                window_mach,         # Mach number (derived from velocity/temp)
                window_pressure,     # Atmospheric pressure (from barometer)
                relative_alt,        # Relative altitude change (from barometer)
                vel_squared,         # Velocity squared (derived)
            ])
            
            samples.append(window_features)
            targets.append(apogee)
    
    print(f"Generated {len(samples)} windows from {len(df)} flights")
    return pd.DataFrame(samples), pd.Series(targets)


# %%
# Generate burnout-phase windows
X_train, y_train = generate_burnout_windows(train_df)
X_test, y_test = generate_burnout_windows(test_df)

# Combine into labeled datasets
train_set = X_train.copy()
train_set["Apogee"] = y_train.values
test_set = X_test.copy()
test_set["Apogee"] = y_test.values

print(f"Training samples: {len(train_set)}, Test samples: {len(test_set)}")

# %%
# Save for future use
model_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
model_dir.mkdir(parents=True, exist_ok=True)

train_model_path = model_dir / "sliding_train_by_flight.csv"
test_model_path = model_dir / "sliding_test_by_flight.csv"

train_set.to_csv(train_model_path, index=False)
test_set.to_csv(test_model_path, index=False)

print(f"Sliding window datasets saved:")
print(f"  Train: {train_model_path}")
print(f"  Test:  {test_model_path}")
