"""
Noise Injection Script (Step 1.5)

Adds realistic sensor noise to clean RocketPy simulation data to better
prepare ML models for real-world flight computer telemetry.

Noise models are calibrated to typical COTS sensors:
  - Accelerometer: BNO055-class (~0.5 m/s² white noise)
  - Barometer: BMP390-class (~1 m altitude noise, ~20 Pa pressure noise)
  - IMU pitch: gyro+accel fusion (~0.5° noise)
  - Integrated quantities: random-walk drift on top of Gaussian noise

Minor outlier spikes are also injected at a low probability per timestep.

Usage:
    python scripts/1.5_noise_injection.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# === PATHS ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "batch_dataset_v1.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "batch_dataset_v1_noisy.csv"
ANALYSIS_DIR = PROJECT_ROOT / "analysis_results"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# === TIMESTEP CONFIG (must match batch simulation) ===
TIMESTEP_INTERVAL = 0.025  # seconds between samples

# === NOISE CONFIGURATION ===
# Each entry: (column_prefix, noise_type, params)
#   noise_type: "gaussian"       -> additive N(0, sigma)
#               "gaussian_drift" -> additive N(0, sigma) + random walk drift
#               "proportional"   -> multiplicative N(1, frac_sigma)
NOISE_CONFIG = {
    "Vertical velocity": {
        "type": "gaussian_drift",
        "sigma": 1.5,           # m/s white noise std
        "drift_intensity": 0.3, # m/s per sqrt(second) random walk
    },
    "Vertical acceleration": {
        "type": "gaussian",
        "sigma": 0.5,           # m/s^2
    },
    "Total velocity": {
        "type": "gaussian_drift",
        "sigma": 1.5,           # m/s
        "drift_intensity": 0.3, # m/s per sqrt(second)
    },
    "Altitude": {
        "type": "gaussian_drift",
        "sigma": 1.0,           # m
        "drift_intensity": 0.2, # m per sqrt(second)
    },
    "Horizontal velocity": {
        "type": "gaussian",
        "sigma": 1.0,           # m/s
    },
    "Pitch angle": {
        "type": "gaussian",
        "sigma": 0.5,           # degrees
    },
    "Dynamic pressure": {
        "type": "proportional",
        "frac_sigma": 0.02,     # 2% of value
    },
    "Mach number": {
        "type": "proportional",
        "frac_sigma": 0.015,    # 1.5% of value
    },
    "Pressure": {
        "type": "gaussian",
        "sigma": 20.0,          # Pa
    },
}

# === OUTLIER CONFIGURATION ===
OUTLIER_PROBABILITY = 0.001   # Probability per timestep per channel (~0.1%)
OUTLIER_MAGNITUDE = 5.0       # Outlier spike = this many sigmas above normal noise

# === REPRODUCIBILITY ===
BASE_SEED = 42


def get_timeseries_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    """Return sorted list of columns matching a telemetry channel prefix."""
    return sorted([c for c in df.columns if c.startswith(prefix)])


def add_gaussian_noise(
    data: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add i.i.d. Gaussian white noise."""
    noise = rng.normal(0, sigma, size=data.shape)
    return data + noise


def add_gaussian_drift_noise(
    data: np.ndarray,
    sigma: float,
    drift_intensity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add Gaussian white noise plus cumulative random-walk drift.

    Drift simulates accelerometer/barometer bias accumulation over time.
    The drift_intensity is in units per sqrt(second).
    """
    n_timesteps = data.shape[1] if data.ndim == 2 else data.shape[0]

    # White noise component
    white_noise = rng.normal(0, sigma, size=data.shape)

    # Random walk drift (per flight row)
    drift_step_sigma = drift_intensity * np.sqrt(TIMESTEP_INTERVAL)
    if data.ndim == 2:
        # Each row is a flight
        drift_steps = rng.normal(0, drift_step_sigma, size=data.shape)
        drift = np.cumsum(drift_steps, axis=1)
    else:
        drift_steps = rng.normal(0, drift_step_sigma, size=data.shape)
        drift = np.cumsum(drift_steps)

    return data + white_noise + drift


def add_proportional_noise(
    data: np.ndarray,
    frac_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add noise proportional to the signal magnitude."""
    noise_factors = rng.normal(1, frac_sigma, size=data.shape)
    return data * noise_factors


def inject_outliers(
    data: np.ndarray,
    noise_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Inject rare spike outliers into the data.

    Each timestep has a small probability of being corrupted by an outlier
    that is several standard deviations away from the normal noise floor.
    """
    outlier_mask = rng.random(size=data.shape) < OUTLIER_PROBABILITY
    outlier_signs = rng.choice([-1, 1], size=data.shape)
    outlier_values = outlier_signs * OUTLIER_MAGNITUDE * noise_sigma
    data_with_outliers = data.copy()
    data_with_outliers[outlier_mask] += outlier_values[outlier_mask]
    return data_with_outliers


def get_effective_sigma(config: dict, data: np.ndarray | None = None) -> float:
    """Get the effective noise sigma for outlier injection scaling."""
    if config["type"] == "proportional":
        # Use a representative signal magnitude for scaling outliers
        if data is not None:
            return config["frac_sigma"] * np.nanmean(np.abs(data))
        return 1.0  # fallback
    return config.get("sigma", 1.0)


def apply_noise_to_channel(
    df: pd.DataFrame,
    prefix: str,
    config: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Apply noise model to all timestep columns of a telemetry channel."""
    cols = get_timeseries_columns(df, prefix)
    if not cols:
        print(f"  ⚠ No columns found for prefix '{prefix}', skipping")
        return df

    data = df[cols].values.copy()

    # Track original NaN positions (post-apogee padding)
    nan_mask = np.isnan(data)

    # Apply channel-specific noise model
    noise_type = config["type"]
    if noise_type == "gaussian":
        noisy_data = add_gaussian_noise(data, config["sigma"], rng)
    elif noise_type == "gaussian_drift":
        noisy_data = add_gaussian_drift_noise(
            data, config["sigma"], config["drift_intensity"], rng
        )
    elif noise_type == "proportional":
        noisy_data = add_proportional_noise(data, config["frac_sigma"], rng)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    # Inject minor outliers
    effective_sigma = get_effective_sigma(config, data)
    noisy_data = inject_outliers(noisy_data, effective_sigma, rng)

    # Restore NaN padding (don't add noise to post-apogee region)
    noisy_data[nan_mask] = np.nan

    df[cols] = noisy_data
    n_outliers = int(np.sum(~nan_mask) * OUTLIER_PROBABILITY)
    print(f"  ✅ {prefix}: {noise_type} noise applied to {len(cols)} timesteps "
          f"(~{n_outliers} outliers injected)")

    return df


def plot_comparison(
    clean_df: pd.DataFrame,
    noisy_df: pd.DataFrame,
    flight_idx: int,
    save_path: Path,
) -> None:
    """Plot clean vs noisy telemetry for a sample flight."""
    channels_to_plot = [
        ("Vertical velocity", "m/s"),
        ("Vertical acceleration", "m/s²"),
        ("Altitude", "m"),
        ("Pressure", "Pa"),
        ("Pitch angle", "deg"),
        ("Dynamic pressure", "Pa"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(
        f"Clean vs Noisy Telemetry — Flight #{flight_idx}",
        fontsize=14,
        fontweight="bold",
    )

    for ax, (prefix, unit) in zip(axes.flat, channels_to_plot):
        cols = get_timeseries_columns(clean_df, prefix)
        # Parse time from column names
        times = []
        for c in cols:
            try:
                t = float(c.split("@ ")[1].replace("s", ""))
                times.append(t)
            except (IndexError, ValueError):
                continue

        if len(times) != len(cols):
            continue

        clean_vals = clean_df.iloc[flight_idx][cols].values.astype(float)
        noisy_vals = noisy_df.iloc[flight_idx][cols].values.astype(float)

        # Only plot up to apogee (non-NaN)
        valid = ~np.isnan(clean_vals)
        times_arr = np.array(times)[valid]
        clean_v = clean_vals[valid]
        noisy_v = noisy_vals[valid]

        ax.plot(times_arr, clean_v, "b-", linewidth=1.5, alpha=0.8, label="Clean")
        ax.plot(times_arr, noisy_v, "r-", linewidth=0.7, alpha=0.6, label="Noisy")
        ax.set_title(f"{prefix} ({unit})", fontsize=11)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(unit)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Comparison plot saved: {save_path}")


def main():
    print("=" * 60)
    print("  Noise Injection — Step 1.5")
    print("=" * 60)

    # Load clean data
    print(f"\nLoading clean dataset: {INPUT_PATH}")
    df_clean = pd.read_csv(INPUT_PATH)
    df_noisy = df_clean.copy()
    print(f"  Flights: {len(df_clean)}, Columns: {df_clean.shape[1]}")

    # Apply noise to each channel
    print("\nApplying sensor noise...")
    rng = np.random.default_rng(BASE_SEED)

    for prefix, config in NOISE_CONFIG.items():
        df_noisy = apply_noise_to_channel(df_noisy, prefix, config, rng)

    # Verify labels are unchanged
    for label_col in ["Apogee altitude (m)", "Apogee time (s)",
                       "Wind Speed (m/s)", "Temperature (K)", "Launch Angle (deg)"]:
        assert (df_clean[label_col] == df_noisy[label_col]).all(), \
            f"Label column '{label_col}' was modified!"
    print("\n✅ Verified: all label/metadata columns unchanged")

    # Compute noise statistics
    print("\nNoise statistics (RMS of added noise):")
    for prefix, config in NOISE_CONFIG.items():
        cols = get_timeseries_columns(df_clean, prefix)
        clean = df_clean[cols].values
        noisy = df_noisy[cols].values
        diff = noisy - clean
        # Ignore NaN positions
        valid_diff = diff[~np.isnan(diff)]
        if len(valid_diff) > 0:
            rms = np.sqrt(np.mean(valid_diff**2))
            print(f"  {prefix}: RMS noise = {rms:.4f}")

    # Save noisy dataset
    df_noisy.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Noisy dataset saved: {OUTPUT_PATH}")
    print(f"   Size: {OUTPUT_PATH.stat().st_size / 1e6:.1f} MB")

    # Plot comparison for a few sample flights
    print("\nGenerating comparison plots...")
    for idx in [0, len(df_clean) // 2, len(df_clean) - 1]:
        plot_path = ANALYSIS_DIR / f"noise_comparison_flight_{idx}.png"
        plot_comparison(df_clean, df_noisy, idx, plot_path)

    print("\n" + "=" * 60)
    print("  Done! Next step: python scripts/2_sliding_window_generator.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
