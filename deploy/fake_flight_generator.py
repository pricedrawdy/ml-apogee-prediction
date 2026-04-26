"""
fake_flight_generator.py — Demo flight data provider for Raspberry Pi

Loads a real (noisy) flight from the RocketPy simulation dataset and
replays it in real-time, producing the exact same feature distribution
the models were trained on.

This approach is strictly better than re-implementing the rocket physics
because it guarantees the feature vectors are in-distribution.

Usage
-----
    from fake_flight_generator import DemoFlight, stream_telemetry, build_inference_window

    demo = DemoFlight(flight_index=0)   # pick any flight 0–N
    for step in demo.stream():
        print(step)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
_RAW  = _REPO / "data" / "raw" / "batch_dataset_v1_noisy.csv"

TIMESTEP         = 0.025    # s — must match training config
WINDOW_SIZE      = int(2.5  / TIMESTEP)   # 100 samples
BURNOUT_END_TIME = 5.5      # s — when we sample the first inference window

# Channels to extract from the dataset
_PREFIXES = [
    "Vertical velocity",
    "Vertical acceleration",
    "Total velocity",
    "Horizontal velocity",
    "Pitch angle",
    "Dynamic pressure",
    "Mach number",
    "Pressure",
    "Altitude",
]


class DemoFlight:
    """
    Wraps one simulated flight from the raw noisy dataset.

    Parameters
    ----------
    flight_index : int
        Row index in the raw CSV (0-based).  Default 0.
    csv_path : Path, optional
        Override path to the raw CSV.  Defaults to data/raw/batch_dataset_v1_noisy.csv.
    """

    def __init__(self, flight_index: int = 0, csv_path: Path | None = None):
        path = csv_path or _RAW
        if not path.exists():
            raise FileNotFoundError(
                f"Raw noisy dataset not found at:\n  {path}\n"
                "Copy it from your dev machine or re-run the noise injection script."
            )

        print(f"[DemoFlight] Loading flight #{flight_index} from {path.name} …", flush=True)
        row = pd.read_csv(path, skiprows=range(1, flight_index + 1), nrows=1)

        self.true_apogee_m: float = float(row["Apogee altitude (m)"].values[0])
        self.burn_time: float = 4.70   # s  (fixed for all flights in dataset)
        self.flight_index = flight_index

        # Extract time-series for each channel
        cols = list(row.columns)
        self._series: dict[str, np.ndarray] = {}
        for prefix in _PREFIXES:
            c = sorted([x for x in cols if x.startswith(prefix)])
            self._series[prefix] = row[c].values[0].astype(float)

        # Trim to apogee (first NaN in vertical velocity)
        vv = self._series["Vertical velocity"]
        nan_idx = np.where(np.isnan(vv))[0]
        self.n_steps = int(nan_idx[0]) if len(nan_idx) else len(vv)

        self.t = np.arange(self.n_steps) * TIMESTEP
        print(f"[DemoFlight] Apogee={self.true_apogee_m:.1f} m  "
              f"({self.true_apogee_m * 3.28084:.0f} ft)  "
              f"flight time={self.t[-1]:.2f} s", flush=True)

    # ── Accessors ─────────────────────────────────────────────────────────────
    def _get(self, prefix: str) -> np.ndarray:
        return self._series[prefix][:self.n_steps]

    @property
    def v_vel(self)    -> np.ndarray: return self._get("Vertical velocity")
    @property
    def v_acc(self)    -> np.ndarray: return self._get("Vertical acceleration")
    @property
    def t_vel(self)    -> np.ndarray: return self._get("Total velocity")
    @property
    def h_vel(self)    -> np.ndarray: return self._get("Horizontal velocity")
    @property
    def pitch(self)    -> np.ndarray: return self._get("Pitch angle")
    @property
    def dynp(self)     -> np.ndarray: return self._get("Dynamic pressure")
    @property
    def mach(self)     -> np.ndarray: return self._get("Mach number")
    @property
    def pressure(self) -> np.ndarray: return self._get("Pressure")
    @property
    def altitude(self) -> np.ndarray: return self._get("Altitude")

    # ── Streaming ─────────────────────────────────────────────────────────────
    def stream(self) -> Iterator[dict]:
        """Yield one telemetry dict per timestep."""
        for i in range(self.n_steps):
            yield {
                "t":        self.t[i],
                "altitude": self.altitude[i],
                "v_vel":    self.v_vel[i],
                "v_acc":    self.v_acc[i],
                "t_vel":    self.t_vel[i],
                "h_vel":    self.h_vel[i],
                "pitch":    self.pitch[i],
                "dynp":     self.dynp[i],
                "mach":     self.mach[i],
                "pressure": self.pressure[i],
                "step_idx": i,
            }

    # ── Feature window ────────────────────────────────────────────────────────
    def build_inference_window(self, window_end_idx: int) -> np.ndarray:
        """
        Build the 1000-element feature vector used during training.

        Feature order (matches 2_sliding_window_generator.py):
            v_vel(100), v_acc(100), t_vel(100), h_vel(100), pitch(100),
            dynp(100), mach(100), pressure(100), relative_alt(100), vel_sq(100)
        """
        start = window_end_idx - WINDOW_SIZE
        pad = 0
        if start < 0:
            pad = -start
            start = 0

        def _slice(arr: np.ndarray) -> np.ndarray:
            s = arr[start:window_end_idx]
            if pad:
                s = np.concatenate([np.zeros(pad), s])
            return s

        w_v_vel  = _slice(self.v_vel)
        w_v_acc  = _slice(self.v_acc)
        w_t_vel  = _slice(self.t_vel)
        w_h_vel  = _slice(self.h_vel)
        w_pitch  = _slice(self.pitch)
        w_dynp   = _slice(self.dynp)
        w_mach   = _slice(self.mach)
        w_press  = _slice(self.pressure)
        w_alt    = _slice(self.altitude)

        relative_alt = w_alt - w_alt[0]
        vel_squared  = w_t_vel ** 2

        return np.concatenate([
            w_v_vel, w_v_acc, w_t_vel, w_h_vel, w_pitch,
            w_dynp, w_mach, w_press, relative_alt, vel_squared,
        ])


# ── Backwards-compat shims (for flight_demo.py) ───────────────────────────────
@dataclass
class FlightProfile:
    """Compatibility shim — parameters used by flight_demo.py."""
    seed:           int   = 42
    flight_index:   int   = 0
    true_apogee_m:  float = 0.0
    burn_time:      float = 4.70
    rocket_mass_kg: float = 7.0
    thrust_newtons: float = 560.0


def create_demo_flight(profile: FlightProfile, csv_path: Path | None = None) -> DemoFlight:
    """Create a DemoFlight from a FlightProfile (uses profile.flight_index)."""
    demo = DemoFlight(flight_index=profile.flight_index, csv_path=csv_path)
    profile.true_apogee_m = demo.true_apogee_m
    profile.burn_time     = demo.burn_time
    return demo
