# Real-time Apogee Prediction Deployment Plan

## Objectives
- Stream live flight telemetry into the apogee prediction models.
- Support multiple model backends (MLP, Random Forest, Linear Regression) for A/B testing and fallback behavior.
- Provide consistent feature scaling and output units aligned with the training pipeline.

## Proposed Architecture
1. **Data Ingestion Layer**
   - Accept telemetry over a message bus (e.g., MQTT or NATS) with per-flight topics.
   - Normalize incoming packets into the same sliding window format used for training (`WINDOW_DURATION`, `STRIDE_DURATION`).
2. **Feature Pipeline Service**
   - Apply the saved `apogee_input_scaler.pkl` to standardized numeric features.
   - Enforce column ordering and fill missing/zero-only fields to maintain model compatibility.
3. **Model Serving Layer**
   - Load the desired model artifact from `models/` (`apogee_mlp_model.pth`, `apogee_random_forest.pkl`, or `apogee_linear_regression.pkl`).
   - Provide a simple RPC or REST endpoint `/predict` that accepts a batch of scaled windows and returns apogee estimates converted via `apogee_target_scaler.pkl`.
   - Include lightweight health endpoints that report model type, commit hash, and scaler versions.
4. **Streaming Evaluator**
   - When truth data becomes available, compute rolling MAE/RMSE to monitor drift.
   - Emit metrics to Prometheus/Grafana and trigger alerts on threshold breaches.
5. **Client Visualization**
   - Extend the existing plotting script to subscribe to live predictions and overlay them with expected apogee bands in near real time.

## Operational Considerations
- **Versioning:** Tag model/scaler pairs together; reject requests when versions mismatch the telemetry schema.
- **Performance:** Cache models in-memory and batch windows to minimize per-request overhead.
- **Fallbacks:** Allow switching between model backends at runtime via configuration to mitigate outages or degradations.
- **Testing:** Build synthetic telemetry replays from recorded flights to validate latency and accuracy before field deployments.

## Next Steps
- Containerize the scaler + model loading code into a lightweight FastAPI/Flask service.
- Define a telemetry message contract (JSON schema) matching the training features.
- Add integration tests that replay `data/processed/sliding_test_by_flight.csv` through the service and compare outputs to offline predictions.
