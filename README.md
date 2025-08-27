# Flood Prediction API

A production-ready Flask API to predict flood risk from real-time sensor inputs.

## Quick start (local)

1. Create a virtual environment and install deps:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Train and export artifacts (model, scaler, config):
```
python train.py
```

3. Run the API in development:
```
python app.py
```

- Test: open http://localhost:5000/test
- Predict (POST JSON to /predict):
```
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"water_level_m\":2.5,\"water_pressure_kpa\":25,\"rain_precip_mm\":1.2}"
```

## Endpoints
- GET `/` basic info
- GET `/health` health and model-loaded status
- GET `/schema` input schema and feature order
- GET `/version` API and artifact presence
- POST `/predict` predict flood risk

## Deploy with Docker
```
docker build -t flood-api .
docker run --rm -p 5000:5000 flood-api
```

## Notes
- Feature order is enforced via `realtime_config.pkl` exported by `train.py`.
- Artifacts required by the API: `realtime_flood_model.pkl`, `realtime_scaler.pkl`, `realtime_config.pkl`.
- For reproducibility, versions are pinned in `requirements.txt`.
