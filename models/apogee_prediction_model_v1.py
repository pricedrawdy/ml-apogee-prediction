"""
Training utilities for Apogee prediction models.

This module now supports training three model variants on the same preprocessed
features/targets:
- A three-layer MLP (baseline)
- A RandomForestRegressor
- A Linear Regression model

Each model is trained on standardized features and targets, and the shared
scalers are persisted for later inference.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Paths
CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "sliding_train_by_flight.csv"
SCALERS_DIR = Path(__file__).resolve().parent.parent / "data" / "scalers"
MODEL_DIR = Path(__file__).resolve().parent
SCALERS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class ApogeeMLP(nn.Module):
    """Three-layer MLP baseline."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x)


def preprocess_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    """Load, clean, and scale the dataset."""
    df = pd.read_csv(CSV_PATH)
    df.fillna(0, inplace=True)

    # Drop columns with all zeros to avoid meaningless features
    df = df.loc[:, (df != 0).any(axis=0)]

    X = df.drop(columns=["Apogee"]).values
    y = df["Apogee"].values.reshape(-1, 1)

    assert not np.isnan(X).any(), "NaN found in features"
    assert not np.isnan(y).any(), "NaN found in targets"

    input_scaler = StandardScaler()
    X_scaled = input_scaler.fit_transform(X)
    joblib.dump(input_scaler, SCALERS_DIR / "apogee_input_scaler.pkl")

    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(y)
    joblib.dump(target_scaler, SCALERS_DIR / "apogee_target_scaler.pkl")

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    return X_train, X_val, y_train, y_val, input_scaler, target_scaler


def train_mlp(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    target_scaler: StandardScaler,
) -> float:
    """Train the three-layer MLP and return validation RMSE in meters."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    train_targets = torch.tensor(y_train, dtype=torch.float32).to(device)
    val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_targets = torch.tensor(y_val, dtype=torch.float32).to(device)

    model = ApogeeMLP(train_tensor.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    batch_size = 64

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(train_tensor.size(0), device=device)
        epoch_loss = 0.0

        for i in range(0, train_tensor.size(0), batch_size):
            indices = permutation[i : i + batch_size]
            batch_x = train_tensor[indices]
            batch_y = train_targets[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_outputs = model(val_tensor)
            val_loss = criterion(val_outputs, val_targets).item()

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

    torch.save(model.state_dict(), MODEL_DIR / "apogee_mlp_model.pth")
    print("✅ Saved MLP model to apogee_mlp_model.pth")

    val_rmse = _inverse_rmse(
        y_val, val_outputs.detach().cpu().numpy().reshape(-1, 1), target_scaler
    )
    return val_rmse


def _inverse_rmse(y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray, target_scaler: StandardScaler) -> float:
    """Compute RMSE in original units using the shared target scaler."""
    y_true = target_scaler.inverse_transform(y_true_scaled.reshape(-1, 1))
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_random_forest(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    target_scaler: StandardScaler,
) -> float:
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
        min_samples_split=2,
    )
    model.fit(X_train, y_train.ravel())

    val_pred_scaled = model.predict(X_val)
    val_rmse = _inverse_rmse(y_val, val_pred_scaled, target_scaler)

    joblib.dump(model, MODEL_DIR / "apogee_random_forest.pkl")
    print("✅ Saved Random Forest model to apogee_random_forest.pkl")

    return val_rmse


def train_linear_regression(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    target_scaler: StandardScaler,
) -> float:
    model = LinearRegression()
    model.fit(X_train, y_train)

    val_pred_scaled = model.predict(X_val)
    val_rmse = _inverse_rmse(y_val, val_pred_scaled, target_scaler)

    joblib.dump(model, MODEL_DIR / "apogee_linear_regression.pkl")
    print("✅ Saved Linear Regression model to apogee_linear_regression.pkl")

    return val_rmse


def main():
    X_train, X_val, y_train, y_val, _, target_scaler = preprocess_data()
    print("Starting training for Apogee models...")

    mlp_rmse = train_mlp(X_train, X_val, y_train, y_val, target_scaler)
    print(f"MLP validation RMSE (meters): {mlp_rmse:.2f}")

    rf_rmse = train_random_forest(X_train, X_val, y_train, y_val, target_scaler)
    print(f"Random Forest validation RMSE (meters): {rf_rmse:.2f}")

    lr_rmse = train_linear_regression(X_train, X_val, y_train, y_val, target_scaler)
    print(f"Linear Regression validation RMSE (meters): {lr_rmse:.2f}")


if __name__ == "__main__":
    main()
