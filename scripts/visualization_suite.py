"""
Comprehensive Visualization Suite for Rocket Apogee Prediction Models.

This script generates a variety of visualizations to demonstrate:
1. Model performance comparison (MLP, Random Forest, Linear Regression)
2. Flight trajectory visualizations (2D and 3D)
3. Additional analytical plots

All outputs are saved to the 'visualizations/' directory.
"""
from __future__ import annotations

import io
import sys
import importlib.util
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === Paths ===
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
SCALERS_DIR = DATA_DIR / "scalers"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "visualizations"

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# === Sliding window/time parameters (must match generator) ===
TIMESTEP_INTERVAL = 0.025
WINDOW_SEC = 2.5
STRIDE_SEC = 0.25
TIMESTEP_INTERVAL = 0.025
WINDOW_SEC = 2.5
STRIDE_SEC = 0.25
TOTAL_TIME_SEC = 25.0

# === Unit Conversion ===
M_TO_FT = 3.28084

WINDOW_SIZE = int(WINDOW_SEC / TIMESTEP_INTERVAL)
STRIDE = int(STRIDE_SEC / TIMESTEP_INTERVAL)
SAMPLES_PER_FLIGHT = int(((TOTAL_TIME_SEC / TIMESTEP_INTERVAL) - WINDOW_SIZE) // STRIDE + 1)

# === Plot style ===
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Color palette for models
MODEL_COLORS = {
    'MLP': '#2ecc71',
    'Random Forest': '#3498db',
    'Linear Regression': '#e74c3c'
}


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_apogee_mlp_class(models_path: Path):
    """Safely import ApogeeMLP class from module."""
    spec = importlib.util.spec_from_file_location(
        "apogee_model_module", str(models_path / "apogee_prediction_model_v1.py")
    )
    module = importlib.util.module_from_spec(spec)
    _stdout, _stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        spec.loader.exec_module(module)
    finally:
        sys.stdout, sys.stderr = _stdout, _stderr
    return module.ApogeeMLP


def load_all_models() -> Dict[str, object]:
    """Load all three model types and scalers."""
    print("Loading models and scalers...")
    
    # Load scalers
    input_scaler = joblib.load(SCALERS_DIR / "apogee_input_scaler.pkl")
    target_scaler = joblib.load(SCALERS_DIR / "apogee_target_scaler.pkl")
    
    # Load MLP
    test_df = pd.read_csv(PROCESSED_DIR / "sliding_test_by_flight.csv")
    input_dim = test_df.shape[1] - 1
    
    ApogeeMLP = load_apogee_mlp_class(MODEL_DIR)
    mlp_model = ApogeeMLP(input_dim=input_dim)
    state = torch.load(MODEL_DIR / "apogee_mlp_model.pth", map_location="cpu")
    mlp_model.load_state_dict(state)
    mlp_model.eval()
    
    # Load Random Forest
    rf_model = joblib.load(MODEL_DIR / "apogee_random_forest.pkl")
    
    # Load Linear Regression
    lr_model = joblib.load(MODEL_DIR / "apogee_linear_regression.pkl")
    
    print("  ✓ All models loaded successfully")
    
    return {
        'MLP': mlp_model,
        'Random Forest': rf_model,
        'Linear Regression': lr_model,
        'input_scaler': input_scaler,
        'target_scaler': target_scaler
    }


def predict_with_model(model, model_name: str, X_scaled: np.ndarray, 
                      target_scaler) -> np.ndarray:
    """Get predictions from a model."""
    if model_name == 'MLP':
        with torch.no_grad():
            y_pred_scaled = model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()
    else:
        y_pred_scaled = model.predict(X_scaled).reshape(-1, 1)
    
    return target_scaler.inverse_transform(y_pred_scaled)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_test_data() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load processed test data."""
    print("Loading test data...")
    test_df = pd.read_csv(PROCESSED_DIR / "sliding_test_by_flight.csv")
    X = test_df.drop(columns=["Apogee"]).values
    y = test_df["Apogee"].values.reshape(-1, 1)
    print(f"  ✓ Loaded {len(test_df)} samples from {len(test_df) // SAMPLES_PER_FLIGHT} flights")
    return test_df, X, y


def load_raw_flight_data() -> pd.DataFrame:
    """Load raw flight trajectory data."""
    print("Loading raw flight data...")
    df = pd.read_csv(RAW_DIR / "batch_dataset_v1.csv")
    print(f"  ✓ Loaded {len(df)} flights with {df.shape[1]} features")
    return df


def extract_trajectory_columns(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Extract time-series trajectory data from raw dataset."""
    cols = df.columns.tolist()
    
    # Extract timesteps from column names
    altitude_cols = [c for c in cols if c.startswith("Altitude (m) @")]
    timesteps = np.array([float(c.split("@")[1].strip().replace("s", "")) for c in altitude_cols])
    
    result = {
        'timesteps': timesteps,
        'altitude': df[[c for c in cols if c.startswith("Altitude (m) @")]].values * M_TO_FT,
        'vertical_velocity': df[[c for c in cols if c.startswith("Vertical velocity (m/s) @")]].values * M_TO_FT,
        'horizontal_velocity': df[[c for c in cols if c.startswith("Horizontal velocity (m/s) @")]].values * M_TO_FT,
        'total_velocity': df[[c for c in cols if c.startswith("Total velocity (m/s) @")]].values * M_TO_FT,
        'mach': df[[c for c in cols if c.startswith("Mach number @")]].values,
        'apogee': df["Apogee altitude (m)"].values * M_TO_FT,
        'apogee_time': df["Apogee time (s)"].values,
        'wind_speed': df["Wind Speed (m/s)"].values * M_TO_FT,
        'temperature': df["Temperature (K)"].values,
        'launch_angle': df["Launch Angle (deg)"].values,
    }
    return result


# ============================================================================
# MODEL PERFORMANCE VISUALIZATIONS
# ============================================================================

def compute_all_model_metrics(models: Dict, X: np.ndarray, y: np.ndarray) -> Dict:
    """Compute metrics for all models."""
    print("Computing model metrics...")
    
    input_scaler = models['input_scaler']
    target_scaler = models['target_scaler']
    X_scaled = input_scaler.transform(X)
    
    metrics = {}
    for name in ['MLP', 'Random Forest', 'Linear Regression']:
        y_pred = predict_with_model(models[name], name, X_scaled, target_scaler)
        
        # Calculate metrics in meters first (standard) then convert for display
        rmse_m = np.sqrt(mean_squared_error(y, y_pred))
        mae_m = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Convert to feet
        rmse_ft = rmse_m * M_TO_FT
        mae_ft = mae_m * M_TO_FT
        y_pred_ft = y_pred * M_TO_FT
        errors_ft = (y_pred - y) * M_TO_FT
        
        metrics[name] = {
            'rmse': rmse_ft,
            'mae': mae_ft,
            'r2': r2,
            'predictions': y_pred_ft,
            'errors': errors_ft
        }
        print(f"  {name}: RMSE={rmse_ft:.2f}ft, MAE={mae_ft:.2f}ft, R²={r2:.4f}")
    
    return metrics


def plot_model_comparison_bar(metrics: Dict):
    """Create bar chart comparing RMSE and MAE across models."""
    print("Generating model comparison bar chart...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    model_names = list(metrics.keys())
    rmse_values = [metrics[m]['rmse'] for m in model_names]
    mae_values = [metrics[m]['mae'] for m in model_names]
    colors = [MODEL_COLORS[m] for m in model_names]
    
    # RMSE bars
    bars1 = axes[0].bar(model_names, rmse_values, color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('RMSE (feet)', fontsize=12)
    axes[0].set_title('Root Mean Square Error by Model', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, max(rmse_values) * 1.15)
    for bar, val in zip(bars1, rmse_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{val:.1f}ft', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # MAE bars
    bars2 = axes[1].bar(model_names, mae_values, color=colors, edgecolor='black', linewidth=1.2)
    axes[1].set_ylabel('MAE (feet)', fontsize=12)
    axes[1].set_title('Mean Absolute Error by Model', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, max(mae_values) * 1.15)
    for bar, val in zip(bars2, mae_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{val:.1f}ft', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "model_comparison_bar.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_pred_vs_actual(metrics: Dict, y_true: np.ndarray):
    """Create scatter plots of predicted vs actual for each model."""
    print("Generating predicted vs actual scatter plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, (name, data) in zip(axes, metrics.items()):
        y_pred = data['predictions']
        color = MODEL_COLORS[name]
        
        ax.scatter(y_true, y_pred, alpha=0.3, s=10, c=color, label='Predictions')
        
        # Perfect prediction line
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, 'k--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Apogee (ft)', fontsize=12)
        ax.set_ylabel('Predicted Apogee (ft)', fontsize=12)
        ax.set_title(f'{name}\nR² = {data["r2"]:.4f}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "pred_vs_actual_scatter.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_error_vs_time(models: Dict, test_df: pd.DataFrame, y_true: np.ndarray):
    """Plot prediction error vs time into flight for all models."""
    print("Generating error vs time comparison...")
    
    input_scaler = models['input_scaler']
    target_scaler = models['target_scaler']
    X = test_df.drop(columns=["Apogee"]).values
    X_scaled = input_scaler.transform(X)
    
    # Time stamps for each sample
    time_per_sample = np.array([
        ((i * STRIDE) + (WINDOW_SIZE / 2.0)) * TIMESTEP_INTERVAL 
        for i in range(SAMPLES_PER_FLIGHT)
    ])
    
    num_flights = len(y_true) // SAMPLES_PER_FLIGHT
    max_time = 10.0  # Limit to first 10 seconds
    time_mask = time_per_sample <= max_time
    filtered_time = time_per_sample[time_mask]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for model_name in ['MLP', 'Random Forest', 'Linear Regression']:
        y_pred = predict_with_model(models[model_name], model_name, X_scaled, target_scaler)
        
        # Compute absolute errors per flight, per timestep
        abs_errors = []
        for flight_idx in range(num_flights):
            start = flight_idx * SAMPLES_PER_FLIGHT
            end = start + SAMPLES_PER_FLIGHT
            
            # Convert to feet for error calculation
            true_apogee = y_true[start][0] * M_TO_FT
            preds_i = y_pred[start:end].flatten()[time_mask] * M_TO_FT
            
            abs_errors.append(np.abs(preds_i - true_apogee))
        
        abs_errors = np.vstack(abs_errors)
        mean_err = np.mean(abs_errors, axis=0)
        std_err = np.std(abs_errors, axis=0)
        
        color = MODEL_COLORS[model_name]
        ax.plot(filtered_time, mean_err, color=color, linewidth=2.5, label=model_name)
        ax.fill_between(filtered_time, 
                        np.maximum(0, mean_err - std_err), 
                        mean_err + std_err, 
                        color=color, alpha=0.15)
    
    ax.set_xlabel('Time into Flight (s)', fontsize=13)
    ax.set_ylabel('Mean Absolute Error (ft)', fontsize=13)
    ax.set_title('Prediction Error vs Flight Time - Model Comparison', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "error_vs_time_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_error_distributions(metrics: Dict):
    """Create violin plots of error distributions for each model."""
    print("Generating error distribution violin plots...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Prepare data for violin plot
    data_list = []
    labels = []
    colors_list = []
    
    for name in ['MLP', 'Random Forest', 'Linear Regression']:
        errors = metrics[name]['errors'].flatten()
        # Sample for visualization (too many points slow down rendering)
        if len(errors) > 5000:
            errors = np.random.choice(errors, 5000, replace=False)
        data_list.append(errors)
        labels.append(name)
        colors_list.append(MODEL_COLORS[name])
    
    parts = ax.violinplot(data_list, positions=[1, 2, 3], showmeans=True, showmedians=True)
    
    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_list[i])
        pc.set_alpha(0.7)
    
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Prediction Error (ft)', fontsize=13)
    ax.set_title('Error Distribution by Model', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "error_distribution_violin.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_feature_importance(models: Dict):
    """Plot feature importance for Random Forest model."""
    print("Generating Random Forest feature importance...")
    
    rf_model = models['Random Forest']
    importances = rf_model.feature_importances_
    
    # Get top 20 features
    top_n = 20
    indices = np.argsort(importances)[-top_n:][::-1]
    top_importances = importances[indices]
    
    # Create simple feature names (indices)
    feature_names = [f"Feature {i}" for i in indices]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(top_importances))
    bars = ax.barh(y_pos, top_importances, color=MODEL_COLORS['Random Forest'], edgecolor='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Most Important Features (Random Forest)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "rf_feature_importance.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


# ============================================================================
# FLIGHT TRAJECTORY VISUALIZATIONS - 2D
# ============================================================================

def plot_2d_all_flights(traj_data: Dict):
    """Plot all flight trajectories (altitude vs time) on one 2D graph."""
    print("Generating 2D all-flights trajectory plot...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    timesteps = traj_data['timesteps']
    altitude = traj_data['altitude']
    apogee_times = traj_data['apogee_time']
    
    num_flights = altitude.shape[0]
    
    # Use colormap based on apogee altitude
    apogees = traj_data['apogee']
    norm = plt.Normalize(apogees.min(), apogees.max())
    cmap = plt.cm.viridis
    
    for i in range(num_flights):
        # Only plot up to apogee time
        mask = timesteps <= apogee_times[i]
        color = cmap(norm(apogees[i]))
        ax.plot(timesteps[mask], altitude[i, mask], alpha=0.4, linewidth=0.8, color=color)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Apogee Altitude (ft)', fontsize=12)
    
    # Add legend with min/max altitude
    min_apogee = apogees.min()
    max_apogee = apogees.max()
    ax.plot([], [], ' ', label=f'Min Apogee: {min_apogee:.0f}ft')
    ax.plot([], [], ' ', label=f'Max Apogee: {max_apogee:.0f}ft')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    ax.set_xlabel('Time (s)', fontsize=13)
    ax.set_ylabel('Altitude (ft)', fontsize=13)
    ax.set_title(f'All Flight Trajectories (n={num_flights})', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 27)
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "2d_all_flights.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_2d_average_flight(traj_data: Dict):
    """Plot average flight trajectory with confidence bands."""
    print("Generating 2D average flight trajectory plot...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    timesteps = traj_data['timesteps']
    altitude = traj_data['altitude']
    
    # Compute mean and std at each timestep
    mean_alt = np.nanmean(altitude, axis=0)
    std_alt = np.nanstd(altitude, axis=0)
    
    # CUTOFF: Truncate when < 10% of flights are active to avoid survivorship bias
    active_counts = np.sum(~np.isnan(altitude), axis=0)
    cutoff_threshold = int(altitude.shape[0] * 0.1)
    cutoff_indices = np.where(active_counts < cutoff_threshold)[0]
    cutoff_idx = cutoff_indices[0] if len(cutoff_indices) > 0 else len(timesteps)
    
    plot_timesteps = timesteps[:cutoff_idx]
    plot_mean_alt = mean_alt[:cutoff_idx]
    plot_std_alt = std_alt[:cutoff_idx]
    
    # Plot mean with confidence bands
    ax.plot(plot_timesteps, plot_mean_alt, color='#2c3e50', linewidth=3, label='Mean Trajectory')
    ax.fill_between(plot_timesteps, 
                    np.maximum(0, plot_mean_alt - plot_std_alt), 
                    plot_mean_alt + plot_std_alt, 
                    color='#3498db', alpha=0.3, label='±1 Std Dev')
    ax.fill_between(plot_timesteps, 
                    np.maximum(0, plot_mean_alt - 2*plot_std_alt), 
                    plot_mean_alt + 2*plot_std_alt, 
                    color='#3498db', alpha=0.15, label='±2 Std Dev')
    
    # Mark mean apogee (using scalar mean of all apogees, not peak of average curve)
    apogees = traj_data['apogee']
    true_mean_apogee = np.mean(apogees)
    mean_apogee_time = np.mean(traj_data['apogee_time'])
    
    ax.scatter([mean_apogee_time], [true_mean_apogee], 
               s=150, c='red', zorder=5, marker='*', label=f'Mean Apogee: {true_mean_apogee:.0f}ft')
    
    # Add min/max altitude to legend
    apogees = traj_data['apogee']
    ax.plot([], [], ' ', label=f'Min Apogee: {apogees.min():.0f}ft')
    ax.plot([], [], ' ', label=f'Max Apogee: {apogees.max():.0f}ft')
    
    ax.set_xlabel('Time (s)', fontsize=13)
    ax.set_ylabel('Altitude (ft)', fontsize=13)
    ax.set_title('Average Flight Trajectory with Confidence Bands', fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 27)
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "2d_average_flight.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


# ============================================================================
# FLIGHT TRAJECTORY VISUALIZATIONS - 3D
# ============================================================================

def compute_3d_positions(traj_data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute approximate X, Y, Z positions from velocity data.
    This is an approximation since we don't have exact position data.
    We integrate horizontal velocity to estimate horizontal displacement.
    """
    timesteps = traj_data['timesteps']
    dt = np.diff(timesteps, prepend=0)
    dt[0] = timesteps[0]
    
    altitude = traj_data['altitude']  # Z
    h_vel = traj_data['horizontal_velocity']
    launch_angles = traj_data['launch_angle']
    
    num_flights = altitude.shape[0]
    num_timesteps = len(timesteps)
    
    # Approximate X, Y from horizontal velocity and launch angle
    # Assume horizontal motion is mostly in the direction of initial tilt
    X = np.zeros((num_flights, num_timesteps))
    Y = np.zeros((num_flights, num_timesteps))
    
    for i in range(num_flights):
        angle_rad = np.radians(launch_angles[i])
        # Integrate horizontal velocity to get horizontal distance
        h_dist = np.cumsum(h_vel[i] * dt)
        X[i] = h_dist * np.cos(angle_rad)
        Y[i] = h_dist * np.sin(angle_rad)
    
    return X, Y, altitude


def plot_3d_all_flights(traj_data: Dict):
    """Plot all flight trajectories in 3D."""
    print("Generating 3D all-flights trajectory plot...")
    
    X, Y, Z = compute_3d_positions(traj_data)
    timesteps = traj_data['timesteps']
    apogee_times = traj_data['apogee_time']
    apogees = traj_data['apogee']
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    num_flights = Z.shape[0]
    norm = plt.Normalize(apogees.min(), apogees.max())
    cmap = plt.cm.plasma
    
    for i in range(num_flights):
        mask = timesteps <= apogee_times[i]
        color = cmap(norm(apogees[i]))
        ax.plot(X[i, mask], Y[i, mask], Z[i, mask], alpha=0.5, linewidth=0.8, color=color)
    
    ax.set_title(f'3D Flight Trajectories (n={num_flights})', fontsize=16, fontweight='bold')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Apogee Altitude (ft)', fontsize=11)
    
    # Add legend with min/max altitude
    min_apogee = apogees.min()
    max_apogee = apogees.max()
    ax.plot([], [], [], ' ', label=f'Min Apogee: {min_apogee:.0f}ft')
    ax.plot([], [], [], ' ', label=f'Max Apogee: {max_apogee:.0f}ft')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    ax.set_xlabel('X Distance (ft)', fontsize=11)
    ax.set_ylabel('Y Distance (ft)', fontsize=11)
    ax.set_zlabel('Altitude (ft)', fontsize=11)
    
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "3d_all_flights.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_3d_average_flight(traj_data: Dict):
    """Plot average flight trajectory in 3D with confidence tube."""
    print("Generating 3D average flight trajectory plot...")
    
    X, Y, Z = compute_3d_positions(traj_data)
    
    # Compute means
    mean_X = np.nanmean(X, axis=0)
    mean_Y = np.nanmean(Y, axis=0)
    mean_Z = np.nanmean(Z, axis=0)
    
    # Standard deviations for confidence tube
    std_X = np.nanstd(X, axis=0)
    std_Y = np.nanstd(Y, axis=0)
    std_Z = np.nanstd(Z, axis=0)
    
    # CUTOFF: Truncate when < 10% of flights are active to avoid survivorship bias
    active_counts = np.sum(~np.isnan(Z), axis=0)
    cutoff_threshold = int(Z.shape[0] * 0.1)
    cutoff_indices = np.where(active_counts < cutoff_threshold)[0]
    cutoff_idx = cutoff_indices[0] if len(cutoff_indices) > 0 else Z.shape[1]
    
    # Truncate arrays
    mean_X = mean_X[:cutoff_idx]
    mean_Y = mean_Y[:cutoff_idx]
    mean_Z = mean_Z[:cutoff_idx]
    std_X = std_X[:cutoff_idx]
    std_Y = std_Y[:cutoff_idx]
    std_Z = std_Z[:cutoff_idx]
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot mean trajectory
    ax.plot(mean_X, mean_Y, mean_Z, color='#e74c3c', linewidth=4, label='Mean Trajectory')
    
    # Plot confidence bounds using scatter (approximation of tube)
    n_points = len(mean_X)
    step = max(1, n_points // 50)  # Sample points for scatter visualization
    
    for i in range(0, n_points, step):
        # Create a small sphere-like scatter at ±1 std
        ax.scatter([mean_X[i] + std_X[i]], [mean_Y[i]], [mean_Z[i]], 
                   c='blue', alpha=0.1, s=20)
        ax.scatter([mean_X[i] - std_X[i]], [mean_Y[i]], [mean_Z[i]], 
                   c='blue', alpha=0.1, s=20)
        ax.scatter([mean_X[i]], [mean_Y[i] + std_Y[i]], [mean_Z[i]], 
                   c='blue', alpha=0.1, s=20)
        ax.scatter([mean_X[i]], [mean_Y[i] - std_Y[i]], [mean_Z[i]], 
                   c='blue', alpha=0.1, s=20)
    
    # Mark mean apogee (using scalar mean of all apogees)
    # We need approximate X/Y for mean apogee time. Interpolate from mean X/Y trajectory
    apogees = traj_data['apogee']
    true_mean_apogee = np.mean(apogees)
    mean_apogee_time = np.mean(traj_data['apogee_time'])
    timesteps = traj_data['timesteps']
    
    # Values at mean apogee time
    apogee_X = np.interp(mean_apogee_time, timesteps[:cutoff_idx], mean_X)
    apogee_Y = np.interp(mean_apogee_time, timesteps[:cutoff_idx], mean_Y)
    
    ax.scatter([apogee_X], [apogee_Y], [true_mean_apogee], 
               s=200, c='gold', marker='*', edgecolors='black', linewidth=1,
               label=f'Mean Apogee: {true_mean_apogee:.0f}ft', zorder=10)
    
    # Add min/max altitude to legend
    apogees = traj_data['apogee']
    ax.plot([], [], [], ' ', label=f'Min Apogee: {apogees.min():.0f}ft')
    ax.plot([], [], [], ' ', label=f'Max Apogee: {apogees.max():.0f}ft')
    
    ax.set_xlabel('X Distance (ft)', fontsize=11)
    ax.set_ylabel('Y Distance (ft)', fontsize=11)
    ax.set_zlabel('Altitude (ft)', fontsize=11)
    ax.set_title('Average 3D Flight Trajectory', fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "3d_average_flight.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


# ============================================================================
# ADDITIONAL VISUALIZATIONS
# ============================================================================

def plot_apogee_distribution(traj_data: Dict):
    """Plot histogram of apogee altitudes."""
    print("Generating apogee distribution histogram...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    apogees = traj_data['apogee']
    
    ax.hist(apogees, bins=30, color='#9b59b6', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(apogees), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(apogees):.0f}ft')
    ax.axvline(np.median(apogees), color='orange', linestyle='--', linewidth=2, 
               label=f'Median: {np.median(apogees):.0f}ft')
    
    ax.set_xlabel('Apogee Altitude (ft)', fontsize=13)
    ax.set_ylabel('Number of Flights', fontsize=13)
    ax.set_title('Distribution of Apogee Altitudes', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "apogee_distribution.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_velocity_profiles(traj_data: Dict):
    """Plot velocity profiles (vertical, horizontal, total) as average."""
    print("Generating velocity profile plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    timesteps = traj_data['timesteps']
    
    # Vertical velocity
    v_vel = traj_data['vertical_velocity']
    mean_v = np.nanmean(v_vel, axis=0)
    std_v = np.nanstd(v_vel, axis=0)
    axes[0].plot(timesteps, mean_v, color='#27ae60', linewidth=2)
    axes[0].fill_between(timesteps, mean_v - std_v, mean_v + std_v, color='#27ae60', alpha=0.2)
    axes[0].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Time (s)', fontsize=12)
    axes[0].set_ylabel('Velocity (ft/s)', fontsize=12)
    axes[0].set_title('Vertical Velocity', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Horizontal velocity
    h_vel = traj_data['horizontal_velocity']
    mean_h = np.nanmean(h_vel, axis=0)
    std_h = np.nanstd(h_vel, axis=0)
    axes[1].plot(timesteps, mean_h, color='#e67e22', linewidth=2)
    axes[1].fill_between(timesteps, np.maximum(0, mean_h - std_h), mean_h + std_h, 
                         color='#e67e22', alpha=0.2)
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('Velocity (ft/s)', fontsize=12)
    axes[1].set_title('Horizontal Velocity', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Total velocity
    t_vel = traj_data['total_velocity']
    mean_t = np.nanmean(t_vel, axis=0)
    std_t = np.nanstd(t_vel, axis=0)
    axes[2].plot(timesteps, mean_t, color='#8e44ad', linewidth=2)
    axes[2].fill_between(timesteps, np.maximum(0, mean_t - std_t), mean_t + std_t, 
                         color='#8e44ad', alpha=0.2)
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].set_ylabel('Velocity (ft/s)', fontsize=12)
    axes[2].set_title('Total Velocity', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "velocity_profiles.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_mach_profile(traj_data: Dict):
    """Plot Mach number profile."""
    print("Generating Mach number profile plot...")
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    timesteps = traj_data['timesteps']
    mach = traj_data['mach']
    
    mean_mach = np.nanmean(mach, axis=0)
    std_mach = np.nanstd(mach, axis=0)
    
    ax.plot(timesteps, mean_mach, color='#c0392b', linewidth=2.5, label='Mean Mach')
    ax.fill_between(timesteps, np.maximum(0, mean_mach - std_mach), mean_mach + std_mach, 
                    color='#c0392b', alpha=0.2, label='±1 Std Dev')
    
    # Add reference lines
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Mach 1 (Sonic)')
    
    ax.set_xlabel('Time (s)', fontsize=13)
    ax.set_ylabel('Mach Number', fontsize=13)
    ax.set_title('Mach Number Profile', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 27)
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "mach_profile.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 60)
    print("COMPREHENSIVE VISUALIZATION SUITE")
    print("=" * 60)
    print()
    
    # Load everything
    models = load_all_models()
    test_df, X_test, y_test = load_test_data()
    raw_df = load_raw_flight_data()
    traj_data = extract_trajectory_columns(raw_df)
    
    print()
    print("-" * 60)
    print("GENERATING MODEL PERFORMANCE VISUALIZATIONS")
    print("-" * 60)
    
    metrics = compute_all_model_metrics(models, X_test, y_test)
    plot_model_comparison_bar(metrics)
    plot_pred_vs_actual(metrics, y_test)
    plot_error_vs_time(models, test_df, y_test)
    plot_error_distributions(metrics)
    plot_feature_importance(models)
    
    print()
    print("-" * 60)
    print("GENERATING 2D FLIGHT TRAJECTORY VISUALIZATIONS")
    print("-" * 60)
    
    plot_2d_all_flights(traj_data)
    plot_2d_average_flight(traj_data)
    
    print()
    print("-" * 60)
    print("GENERATING 3D FLIGHT TRAJECTORY VISUALIZATIONS")
    print("-" * 60)
    
    plot_3d_all_flights(traj_data)
    plot_3d_average_flight(traj_data)
    
    print()
    print("-" * 60)
    print("GENERATING ADDITIONAL VISUALIZATIONS")
    print("-" * 60)
    
    plot_apogee_distribution(traj_data)
    plot_velocity_profiles(traj_data)
    plot_mach_profile(traj_data)
    
    print()
    print("=" * 60)
    print("VISUALIZATION SUITE COMPLETE")
    print("=" * 60)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  • {f.name}")


if __name__ == "__main__":
    main()
