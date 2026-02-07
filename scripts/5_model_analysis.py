import io
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import joblib
import importlib.util
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

# === Paths ===
root_dir = Path(__file__).resolve().parent.parent
scalers_dir = root_dir / "data" / "scalers"
model_dir = root_dir / "models"
data_dir = root_dir / "data" / "processed"

# === Load scalers ===
input_scaler = joblib.load(scalers_dir / "apogee_input_scaler.pkl")
target_scaler = joblib.load(scalers_dir / "apogee_target_scaler.pkl")

# === Safely import model class (avoid Unicode print crash) ===
def load_apogee_mlp_class(models_path: Path):
    spec = importlib.util.spec_from_file_location(
        "apogee_model_module", str(root_dir / "scripts" / "3_model_creation.py")
    )
    module = importlib.util.module_from_spec(spec)
    # Silence stdout/stderr during import to avoid encoding errors on Windows console
    _stdout, _stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    finally:
        sys.stdout, sys.stderr = _stdout, _stderr
    return module.ApogeeMLP

ApogeeMLP = load_apogee_mlp_class(model_dir)

# === Get input dimension from test data ===
test_df_temp = pd.read_csv(data_dir / "sliding_test_by_flight.csv")
input_dim = test_df_temp.shape[1] - 1  # Subtract 1 for target column
del test_df_temp

# === Load weights ===
model = ApogeeMLP(input_dim=input_dim)
state = torch.load(model_dir / "apogee_mlp_model.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()

# === Sliding window/time parameters (must match your generator) ===
timestep_interval = 0.025
window_sec = 2.5
stride_sec = 0.25
total_time_sec = 25.0  # total simulated duration used to build windows

window_size = int(window_sec / timestep_interval)
stride = int(stride_sec / timestep_interval)
samples_per_flight = int(((total_time_sec / timestep_interval) - window_size) // stride + 1)

# For analysis/plots, you can cap to early flight if desired
max_flight_time = 10.0  # seconds to analyze/plot

def _predict_from_df(df: pd.DataFrame):
    X = df.drop(columns=["Apogee"]).values
    y_true = df["Apogee"].values.reshape(-1, 1)
    X_scaled = input_scaler.transform(X)
    with torch.no_grad():
        y_pred_scaled = model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    return y_true, y_pred

# === 1) Overfitting check (train vs test) ===
def check_overfitting():
    train_df = pd.read_csv(data_dir / "sliding_train_by_flight.csv")
    test_df = pd.read_csv(data_dir / "sliding_test_by_flight.csv")

    y_train_true, y_train_pred = _predict_from_df(train_df)
    y_test_true, y_test_pred = _predict_from_df(test_df)

    train_rmse = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    train_mae = mean_absolute_error(y_train_true, y_train_pred)
    test_mae = mean_absolute_error(y_test_true, y_test_pred)

    print("=== Overfitting Analysis ===")
    print(f"Training RMSE: {train_rmse:.2f} m")
    print(f"Testing  RMSE: {test_rmse:.2f} m")
    print(f"RMSE gap:      {test_rmse - train_rmse:.2f} m")
    print(f"Training MAE:  {train_mae:.2f} m")
    print(f"Testing  MAE:  {test_mae:.2f} m")
    print(f"MAE gap:       {test_mae - train_mae:.2f} m")

    # FIX: return the test arrays computed above
    return test_df, y_test_true, y_test_pred

def assert_flight_blocking(df: pd.DataFrame):
    # Sanity check: apogee should be constant within each flight block
    problems = []
    n = len(df)
    num_blocks = n // samples_per_flight
    vals = df["Apogee"].values
    for b in range(num_blocks):
        s = b * samples_per_flight
        e = s + samples_per_flight
        block = vals[s:e]
        if np.nanstd(block) > 1e-3:  # tweak tolerance if needed
            problems.append(b)
    if problems:
        print(f"WARNING: {len(problems)} flight blocks have varying Apogee (e.g., shuffled rows).")
        print(f"Blocks: {problems[:10]}{'...' if len(problems) > 10 else ''}")
    else:
        print("Flight blocking looks consistent (constant Apogee per block).")
# luke is a big dum dum
# === 2) Multi-flight performance visualization ===
def analyze_flight_predictions(test_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray):
    assert_flight_blocking(test_df)

    # Time stamps are the center of each window
    time_per_sample = np.array(
        [((i * stride) + (window_size / 2.0)) * timestep_interval for i in range(samples_per_flight)]
    )
    time_mask = time_per_sample <= max_flight_time
    filtered_time = time_per_sample[time_mask]

    # Determine number of complete flights represented
    num_flights = len(y_true) // samples_per_flight

    abs_errors = []
    rel_errors = []

    for flight_idx in tqdm(range(num_flights), desc="Processing flights"):
        start = flight_idx * samples_per_flight
        end = start + samples_per_flight

        preds_i = y_pred[start:end].flatten()
        true_apogee = y_true[start][0]  # same target per flight across windows

        # Restrict to time window for analysis
        preds_i = preds_i[time_mask]

        abs_err_i = np.abs(preds_i - true_apogee)
        rel_err_i = abs_err_i / max(true_apogee, 1e-6) * 100.0  # percent

        abs_errors.append(abs_err_i)
        rel_errors.append(rel_err_i)

    abs_errors = np.vstack(abs_errors) if abs_errors else np.empty((0, len(filtered_time)))
    rel_errors = np.vstack(rel_errors) if rel_errors else np.empty((0, len(filtered_time)))

    # Mean/Std over flights
    mean_abs_error = np.mean(abs_errors, axis=0)
    std_abs_error = np.std(abs_errors, axis=0)
    mean_rel_error = np.mean(rel_errors, axis=0)
    std_rel_error = np.std(rel_errors, axis=0)

    # Convergence analysis: within threshold %
    convergence_threshold = 5.0  # percent
    within_threshold = rel_errors <= convergence_threshold
    pct_converged = np.mean(within_threshold, axis=0) * 100.0

    # Time to sustained convergence (3 consecutive points)
    time_to_convergence = []
    for i in range(within_threshold.shape[0]):
        idxs = np.where(within_threshold[i])[0]
        if len(idxs) >= 3:
            for k in range(len(idxs) - 2):
                if idxs[k + 2] - idxs[k] == 2:  # three consecutive
                    time_to_convergence.append(filtered_time[idxs[k]])
                    break

    # Also compute signed error to see bias
    signed_errors = []
    for flight_idx in range(len(y_true) // samples_per_flight):
        s = flight_idx * samples_per_flight
        e = s + samples_per_flight
        preds_i = y_pred[s:e].flatten()[time_mask]
        true_apogee = y_true[s][0]
        signed_errors.append(preds_i - true_apogee)
    signed_errors = np.vstack(signed_errors) if signed_errors else np.empty((0, len(filtered_time)))
    mean_signed = np.mean(signed_errors, axis=0)
    std_signed = np.std(signed_errors, axis=0)

    # === Plots ===

    # MAE with non-negative band
    plt.figure(figsize=(12, 6))
    lower = np.maximum(0, mean_abs_error - std_abs_error)
    upper = mean_abs_error + std_abs_error
    plt.plot(filtered_time, mean_abs_error, 'b-', label='Mean Absolute Error')
    plt.fill_between(filtered_time, lower, upper, color='b', alpha=0.2, label='±1 Std')
    plt.title("Apogee Prediction Mean Absolute Error vs Time")
    plt.xlabel("Time into Flight (s)")
    plt.ylabel("Absolute Error (m)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(root_dir / "apogee_mean_abs_error.png", dpi=300)
    plt.show()

    # Signed bias over time
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_time, mean_signed, 'm-', label='Mean Signed Error (Pred - True)')
    plt.fill_between(filtered_time, mean_signed - std_signed, mean_signed + std_signed, color='m', alpha=0.15)
    plt.axhline(0, color='k', lw=1)
    plt.title("Apogee Prediction Bias vs Time")
    plt.xlabel("Time into Flight (s)")
    plt.ylabel("Signed Error (m)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(root_dir / "apogee_mean_signed_error.png", dpi=300)
    plt.show()

    if len(time_to_convergence) > 0:
        plt.figure(figsize=(12, 6))
        sns.histplot(time_to_convergence, bins=15, kde=True)
        median_t = float(np.median(time_to_convergence))
        plt.axvline(median_t, color='r', linestyle='--', label=f"Median: {median_t:.2f}s")
        plt.title(f"Time to Convergence (<= {convergence_threshold}% error)")
        plt.xlabel("Time into Flight (s)")
        plt.ylabel("Flights")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(root_dir / "apogee_convergence_distribution.png", dpi=300)
        plt.show()

    # Optional: heatmap of relative error (cap flights shown for readability)
    max_flights_to_show = min(50, rel_errors.shape[0])
    if max_flights_to_show > 0:
        plt.figure(figsize=(14, 8))
        sns.heatmap(rel_errors[:max_flights_to_show],
                    cmap='viridis_r',
                    vmax=20,
                    cbar_kws={'label': 'Relative Error (%)'},
                    xticklabels=[f"{t:.1f}" for t in filtered_time])
        plt.title("Relative Error by Flight and Time")
        plt.xlabel("Time into Flight (s)")
        plt.ylabel("Flight #")
        plt.tight_layout()
        plt.savefig(root_dir / "apogee_error_heatmap.png", dpi=300)
        plt.show()

    # === Text summary ===
    print("\n=== Convergence Summary ===")
    print(f"Flights analyzed: {rel_errors.shape[0]}")
    if len(time_to_convergence) > 0:
        print(f"Mean time to convergence:   {np.mean(time_to_convergence):.2f}s")
        print(f"Median time to convergence: {np.median(time_to_convergence):.2f}s")
    else:
        print("No flights reached the convergence criterion.")
    print(f"Converged by end (%):       {pct_converged[-1]:.1f}%")
    # Example checkpoint at ~3s
    idx_3s = int(np.argmin(np.abs(filtered_time - 3.0)))
    print(f"Converged by t=3s (%):      {pct_converged[idx_3s]:.1f}%")

if __name__ == "__main__":
    # 1) Overfitting check
    test_df, y_true_test, y_pred_test = check_overfitting()

    # 2) Aggregate multi-flight performance
    analyze_flight_predictions(test_df, y_true_test, y_pred_test)