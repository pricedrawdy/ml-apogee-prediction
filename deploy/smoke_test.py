"""
smoke_test.py — Quick validation that all deploy components work.

Run from the repo root:
    python deploy/smoke_test.py

Or on the Pi after setup_pi.sh:
    python deploy/smoke_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from fake_flight_generator import DemoFlight, WINDOW_SIZE
from inference_engine import InferenceEngine

TRIGGER_T = 5.5   # seconds — matches burnout inference window midpoint
MODELS    = ["mlp", "rf", "lr"]

def main():
    print("=" * 55)
    print("  RocketPy Deploy — Smoke Test")
    print("=" * 55)

    demo = DemoFlight(flight_index=0)
    engine = InferenceEngine()

    # Stream until inference trigger
    history = []
    feat = None
    for step in demo.stream():
        history.append(step)
        if step["t"] >= TRIGGER_T:
            win = history[-WINDOW_SIZE:]
            pad = WINDOW_SIZE - len(win)
            def _col(key):
                arr = np.array([h[key] for h in win], dtype=float)
                return np.concatenate([np.zeros(pad), arr]) if pad else arr
            w_vv=_col("v_vel"); w_va=_col("v_acc"); w_tv=_col("t_vel")
            w_hv=_col("h_vel"); w_pi=_col("pitch"); w_dq=_col("dynp")
            w_ma=_col("mach");  w_pr=_col("pressure"); w_al=_col("altitude")
            feat = np.concatenate([w_vv,w_va,w_tv,w_hv,w_pi,w_dq,w_ma,w_pr,
                                    w_al-w_al[0], w_tv**2])
            break

    if feat is None:
        print("FAIL: could not build inference window")
        sys.exit(1)

    print(f"\nFlight #0  |  True apogee: {demo.true_apogee_m:.1f} m  ({demo.true_apogee_m*3.28084:.0f} ft)")
    print(f"Inference window: t={TRIGGER_T:.1f}s  |  Feature dim: {len(feat)}\n")

    all_ok = True
    for model in MODELS:
        engine.load(model)
        result = engine.predict(feat)
        err    = result.apogee_m - demo.true_apogee_m
        err_pc = err / demo.true_apogee_m * 100
        ok     = abs(err_pc) < 10.0
        status = "OK  " if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{status}] {model.upper():3s}  "
              f"predicted={result.apogee_m:.1f}m  "
              f"err={err:+.1f}m ({err_pc:+.1f}%)  "
              f"{result.elapsed_ms:.1f}ms")

    print()
    print("ALL PASS" if all_ok else "SOME TESTS FAILED")
    print("=" * 55)
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()
