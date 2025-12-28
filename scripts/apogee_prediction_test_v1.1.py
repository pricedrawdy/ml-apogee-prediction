import argparse
from pathlib import Path
from typing import Dict, Optional

import joblib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error

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
    parser.add_argument(
        "--plot-max-time",
        type=float,
        default=DEFAULT_PLOT_MAX_TIME,
        help="Seconds to display; set <=0 to show the full flight window.",
    )
    parser.add_argument("--timestep", type=float, default=DEFAULT_TIMESTEP)
    parser.add_argument("--window-duration", type=float, default=DEFAULT_WINDOW_DURATION)
    parser.add_argument("--stride-duration", type=float, default=DEFAULT_STRIDE_DURATION)
    parser.add_argument("--total-flight-time", type=float, default=DEFAULT_TOTAL_FLIGHT_TIME)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip opening an interactive plot window (useful when importing).",
    )
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


def _build_context(
    timestep: float,
    window_duration: float,
    stride_duration: float,
    total_flight_time: float,
) -> Dict:
    base_dir = Path(__file__).resolve().parent.parent
    scalers_dir = base_dir / "data" / "scalers"
    model_dir = base_dir / "models"
    data_dir = base_dir / "data" / "processed"

    window_size = int(window_duration / timestep)
    stride = int(stride_duration / timestep)
    samples_per_flight = int(((total_flight_time / timestep - window_size) // stride) + 1)
    time_axis = np.array([((i * stride) + window_size // 2) * timestep for i in range(samples_per_flight)])

    input_scaler = joblib.load(scalers_dir / "apogee_input_scaler.pkl")
    target_scaler = joblib.load(scalers_dir / "apogee_target_scaler.pkl")

    df = pd.read_csv(data_dir / "sliding_test_by_flight.csv")
    X = df.drop(columns=["Apogee"]).values
    y_true = df["Apogee"].values.reshape(-1, 1)

    return {
        "base_dir": base_dir,
        "model_dir": model_dir,
        "df": df,
        "X": X,
        "y_true": y_true,
        "input_scaler": input_scaler,
        "target_scaler": target_scaler,
        "window_size": window_size,
        "stride": stride,
        "samples_per_flight": samples_per_flight,
        "time_axis": time_axis,
    }


def evaluate_model(
    model_type: str,
    flight_index: int = DEFAULT_FLIGHT_INDEX,
    plot_max_time: Optional[float] = DEFAULT_PLOT_MAX_TIME,
    timestep: float = DEFAULT_TIMESTEP,
    window_duration: float = DEFAULT_WINDOW_DURATION,
    stride_duration: float = DEFAULT_STRIDE_DURATION,
    total_flight_time: float = DEFAULT_TOTAL_FLIGHT_TIME,
    context: Optional[Dict] = None,
    show_plot: bool = False,
) -> Dict:
    ctx = context or _build_context(
        timestep=timestep,
        window_duration=window_duration,
        stride_duration=stride_duration,
        total_flight_time=total_flight_time,
    )

    model = load_model(model_type, input_dim=ctx["X"].shape[1], model_dir=ctx["model_dir"])

    X_scaled = ctx["input_scaler"].transform(ctx["X"])
    y_pred_scaled = run_inference(model_type, model, X_scaled)
    y_pred = ctx["target_scaler"].inverse_transform(y_pred_scaled)

    mse = mean_squared_error(ctx["y_true"], y_pred)
    rmse = float(np.sqrt(mse))

    start_idx = flight_index * ctx["samples_per_flight"]
    end_idx = start_idx + ctx["samples_per_flight"]

    if start_idx >= len(y_pred):
        raise ValueError(f"Flight index {flight_index} out of bounds.")

    flight_preds = y_pred[start_idx:end_idx].flatten()
    flight_true = ctx["y_true"][start_idx:end_idx].flatten()
    target_apogee = float(flight_true[0])  # constant apogee per flight

    error = np.abs(flight_preds - target_apogee)
    mean_abs_error = float(np.mean(error))
    max_error = float(np.max(error))

    use_full_window = plot_max_time is None or plot_max_time <= 0
    if use_full_window:
        masked_time = ctx["time_axis"]
        masked_preds = flight_preds
    else:
        mask = ctx["time_axis"] <= plot_max_time
        masked_time = ctx["time_axis"][mask]
        masked_preds = flight_preds[mask]
        if len(masked_time) == 0:
            masked_time = ctx["time_axis"]
            masked_preds = flight_preds

    # Avoid pyplot when used from a worker thread (e.g., Tkinter) to prevent GUI backend issues
    if show_plot:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
    else:
        fig = Figure(figsize=(7.5, 4.2))
        ax = fig.add_subplot(111)
    ax.plot(masked_time, masked_preds, "b-o", label="Predicted Apogee", markersize=4)
    ax.hlines(target_apogee, masked_time[0], masked_time[-1], colors="r", linestyles="--", label="True Apogee")

    error_margin = 0.01 * target_apogee
    ax.fill_between(
        masked_time,
        target_apogee - error_margin,
        target_apogee + error_margin,
        color="r",
        alpha=0.1,
        label="±1% Margin",
    )

    ax.set_title(f"Apogee Prediction: Flight {flight_index} ({model_type})")
    ax.set_xlabel("Time into Flight (s)")
    ax.set_ylabel("Apogee (m)")
    ax.set_xlim(min(masked_time), max(masked_time))
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if show_plot:
        plt.show(block=True)

    return {
        "figure": fig,
        "rmse": rmse,
        "mean_abs_error": mean_abs_error,
        "max_error": max_error,
        "target_apogee": target_apogee,
        "flight_index": flight_index,
        "model_type": model_type,
    }


def main():
    args = parse_args()

    plot_time = None if args.plot_max_time <= 0 else args.plot_max_time

    ctx = _build_context(
        timestep=args.timestep,
        window_duration=args.window_duration,
        stride_duration=args.stride_duration,
        total_flight_time=args.total_flight_time,
    )

    print(f"Loading model '{args.model}' and evaluating flight {args.flight_index} ...")
    result = evaluate_model(
        args.model,
        flight_index=args.flight_index,
        plot_max_time=plot_time,
        timestep=args.timestep,
        window_duration=args.window_duration,
        stride_duration=args.stride_duration,
        total_flight_time=args.total_flight_time,
        context=ctx,
        show_plot=not args.no_show,
    )

    print(f"\n=== Analysis for Flight {args.flight_index} ===")
    print(f"True Apogee: {result['target_apogee']:.2f} m")
    print(f"Mean Absolute Error: {result['mean_abs_error']:.2f} m")
    print(f"Max Error: {result['max_error']:.2f} m")
    print(f"Global RMSE: {result['rmse']:.2f} meters")


if __name__ == "__main__":
    main()
