import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# === Default configuration ===
DEFAULT_FLIGHT_INDEX = 2            # Flight to analyze/plot (0-based)
DEFAULT_TIMESTEP = 0.025            # Seconds
DEFAULT_WINDOW_DURATION = 2.5       # Seconds
DEFAULT_STRIDE_DURATION = 0.25      # Seconds
DEFAULT_TOTAL_FLIGHT_TIME = 25.0    # Seconds (Total duration per flight in dataset)
DEFAULT_PLOT_MAX_TIME = 10.0        # Seconds (Limit x-axis for visibility)

MODEL_FILENAMES = {
    "mlp": "apogee_mlp_model.pth",
    "random_forest": "apogee_random_forest.pkl",
    "linear_regression": "apogee_linear_regression.pkl",
}


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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Apogee prediction models")
    parser.add_argument(
        "--model",
        choices=list(MODEL_FILENAMES.keys()),
        default="mlp",
        help="Which trained model to evaluate",
    )
    parser.add_argument("--flight-index", type=int, default=DEFAULT_FLIGHT_INDEX)
    parser.add_argument("--plot-max-time", type=float, default=DEFAULT_PLOT_MAX_TIME)
    parser.add_argument("--timestep", type=float, default=DEFAULT_TIMESTEP)
    parser.add_argument("--window-duration", type=float, default=DEFAULT_WINDOW_DURATION)
    parser.add_argument("--stride-duration", type=float, default=DEFAULT_STRIDE_DURATION)
    parser.add_argument("--total-flight-time", type=float, default=DEFAULT_TOTAL_FLIGHT_TIME)
    return parser.parse_args()


def load_model(model_type: str, input_dim: int, model_dir: Path):
    model_path = model_dir / MODEL_FILENAMES[model_type]
    if model_type == "mlp":
        model = ApogeeMLP(input_dim=input_dim)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model

    return joblib.load(model_path)


def run_inference(model_type: str, model, X_scaled: np.ndarray) -> np.ndarray:
    if model_type == "mlp":
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            preds_scaled = model(X_tensor).numpy()
    else:
        preds_scaled = model.predict(X_scaled).reshape(-1, 1)
    return preds_scaled


def main():
    args = parse_args()

    # Calculated parameters
    window_size = int(args.window_duration / args.timestep)
    stride = int(args.stride_duration / args.timestep)
    samples_per_flight = int(((args.total_flight_time / args.timestep - window_size) // stride) + 1)

    base_dir = Path(__file__).resolve().parent.parent
    scalers_dir = base_dir / "data" / "scalers"
    model_dir = base_dir / "models"
    data_dir = base_dir / "data" / "processed"

    print("Loading model and scalers...")
    input_scaler = joblib.load(scalers_dir / "apogee_input_scaler.pkl")
    target_scaler = joblib.load(scalers_dir / "apogee_target_scaler.pkl")

    df = pd.read_csv(data_dir / "sliding_test_by_flight.csv")
    X = df.drop(columns=["Apogee"]).values
    y_true = df["Apogee"].values.reshape(-1, 1)

    model = load_model(args.model, input_dim=X.shape[1], model_dir=model_dir)

    print(f"Running inference with {args.model} model...")
    X_scaled = input_scaler.transform(X)
    y_pred_scaled = run_inference(args.model, model, X_scaled)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"Global RMSE: {rmse:.2f} meters")

    start_idx = args.flight_index * samples_per_flight
    end_idx = start_idx + samples_per_flight

    if start_idx >= len(y_pred):
        raise ValueError(f"Flight index {args.flight_index} out of bounds.")

    flight_preds = y_pred[start_idx:end_idx].flatten()
    flight_true = y_true[start_idx:end_idx].flatten()
    target_apogee = flight_true[0]  # Assuming constant apogee per flight

    time_axis = np.array([((i * stride) + window_size // 2) * args.timestep
                          for i in range(len(flight_preds))])

    error = np.abs(flight_preds - target_apogee)
    print(f"\n=== Analysis for Flight {args.flight_index} ===")
    print(f"True Apogee: {target_apogee:.2f} m")
    print(f"Mean Absolute Error: {np.mean(error):.2f} m")
    print(f"Max Error: {np.max(error):.2f} m")

    mask = time_axis <= args.plot_max_time

    plt.figure(figsize=(12, 6))
    plt.plot(time_axis[mask], flight_preds[mask], 'b-o', label='Predicted Apogee', markersize=4)
    plt.hlines(target_apogee, time_axis[0], args.plot_max_time, colors='r', linestyles='--', label='True Apogee')

    error_margin = 0.01 * target_apogee
    plt.fill_between(time_axis[mask],
                     target_apogee - error_margin,
                     target_apogee + error_margin,
                     color='r', alpha=0.1, label='±1% Margin')

    plt.title(f"Apogee Prediction: Flight {args.flight_index} ({args.model})")
    plt.xlabel("Time into Flight (s)")
    plt.ylabel("Apogee (m)")
    plt.xlim(0, args.plot_max_time)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
