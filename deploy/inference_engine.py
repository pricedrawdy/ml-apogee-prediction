"""
inference_engine.py — Lightweight single-window inference for Raspberry Pi

Loads one (or all) trained models from the models/ directory and runs
# encoding: utf-8
inference on a single feature vector built by build_inference_window().

No sliding window CSV is needed — just the saved .pkl / .pth files and scalers.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Literal

# Suppress sklearn version mismatch warning (scalers saved with different version)
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")

import joblib
import numpy as np
import torch
import torch.nn as nn

# ── Paths (relative to this file which lives in deploy/) ─────────────────────
_REPO = Path(__file__).resolve().parent.parent
MODEL_DIR  = _REPO / "models"
SCALER_DIR = _REPO / "data" / "scalers"

ModelType = Literal["mlp", "rf", "lr"]


# ── MLP definition (must match 3_model_creation.py exactly) ──────────────────
class ApogeeMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class InferenceEngine:
    """
    Wraps all three apogee models and the shared scalers.

    Usage
    -----
        engine = InferenceEngine()
        engine.load("mlp")            # or "rf" / "lr"
        result = engine.predict(feature_vector_np)
        print(f"Predicted apogee: {result.apogee_m:.1f} m  ({result.elapsed_ms:.0f} ms)")
    """

    def __init__(self):
        self._input_scaler  = None
        self._target_scaler = None
        self._model         = None
        self._model_type: ModelType | None = None
        self._input_dim: int | None = None

    # ── Load ──────────────────────────────────────────────────────────────────
    def load(self, model_type: ModelType = "mlp") -> None:
        """Load scalers + model.  Call once at startup."""
        print(f"[Inference] Loading scalers …")
        self._input_scaler  = joblib.load(SCALER_DIR / "apogee_input_scaler.pkl")
        self._target_scaler = joblib.load(SCALER_DIR / "apogee_target_scaler.pkl")

        print(f"[Inference] Loading model: {model_type}")
        if model_type == "mlp":
            # Determine input_dim from scaler
            self._input_dim = self._input_scaler.n_features_in_
            model = ApogeeMLP(self._input_dim)
            state = torch.load(
                MODEL_DIR / "apogee_mlp_model.pth",
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state)
            model.eval()
            self._model = model

        elif model_type == "rf":
            self._model = joblib.load(MODEL_DIR / "apogee_random_forest.pkl")

        elif model_type == "lr":
            self._model = joblib.load(MODEL_DIR / "apogee_linear_regression.pkl")

        else:
            raise ValueError(f"Unknown model type: {model_type!r}")

        self._model_type = model_type
        print(f"[Inference] OK  Ready  (model={model_type})")

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict(self, features: np.ndarray) -> "PredictionResult":
        """
        Run inference on a single feature window.

        Parameters
        ----------
        features : np.ndarray, shape (input_dim,)

        Returns
        -------
        PredictionResult
        """
        if self._model is None:
            raise RuntimeError("Call load() before predict()")

        x = features.reshape(1, -1)
        x_scaled = self._input_scaler.transform(x)

        t0 = time.perf_counter()
        if self._model_type == "mlp":
            with torch.no_grad():
                tensor = torch.tensor(x_scaled, dtype=torch.float32)
                pred_scaled = self._model(tensor).numpy()
        else:
            pred_scaled = self._model.predict(x_scaled).reshape(-1, 1)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        apogee_m = float(self._target_scaler.inverse_transform(pred_scaled)[0, 0])

        return PredictionResult(
            apogee_m   = apogee_m,
            apogee_ft  = apogee_m * 3.28084,
            elapsed_ms = elapsed_ms,
            model_type = self._model_type,
        )


# ── Result dataclass ──────────────────────────────────────────────────────────
from dataclasses import dataclass

@dataclass
class PredictionResult:
    apogee_m:   float
    apogee_ft:  float
    elapsed_ms: float
    model_type: str

    def __str__(self) -> str:
        return (
            f"Predicted Apogee: {self.apogee_m:.1f} m  "
            f"({self.apogee_ft:.0f} ft)  "
            f"[{self.model_type}  {self.elapsed_ms:.1f} ms]"
        )
