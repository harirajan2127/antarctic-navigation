# API

Two HTTP applications serve the system. Both expose OpenAPI `/docs`
(Swagger) and `/openapi.json`. Responses that use demo data are labeled
`demo=true` with a `warning`; models report `model_used_real` only when
trained on real observations.

## Unified REST API — `uvicorn main:app` (prefix `/api`, port 8000)

Primary surface for the dashboard (`frontend/` Vite proxy → `/api` → :8000).

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Health + DB status |
| GET | `/api/datasets` | Dataset inventory with real/demo classification |
| GET | `/api/datasets/status` | Per-dataset status + aggregate `real_data_available` |
| GET | `/api/sea-ice/current` | Current sea-ice concentration field |
| GET | `/api/sea-ice/forecast?horizon_hours=&model=` | Sea-ice forecast |
| POST | `/api/sea-ice/predict` | Forecast from optional custom inputs |
| GET | `/api/icebergs` | List tracked icebergs |
| GET | `/api/icebergs/{iceberg_id}` | Iceberg detail |
| GET | `/api/icebergs/{iceberg_id}/trajectory` | History + predicted trajectory |
| POST | `/api/icebergs/predict` | Predict positions at a horizon |
| GET | `/api/icebergs/distance?iceberg_a=&iceberg_b=` | km + nm distance between two icebergs |
| POST | `/api/routes/optimize` | Recommended route + alternatives (persisted → `route_id`) |
| GET | `/api/routes/{route_id}` | Retrieve a stored route |
| GET | `/api/analytics/summary` | Dashboard summary (incl. `real_data_available`) |
| GET | `/api/analytics/model-metrics` | Seeded trained-model validation metrics |
| GET | `/api/models/status` | Trained-model registry (chip availability, demo/real) |
| GET | `/api/vessels` | Vessel catalog |
| POST | `/api/assistant/chat` | Tool-backed chat (`{question, history, horizon_hours}`) |

NOTE: static `/api/icebergs/distance` and `/api/icebergs/predict` are
registered before `/{iceberg_id}`.

## Legacy journey API — `uvicorn app.main:app` (prefix `/api/v1`, port 8001)

Owns the journey lifecycle that drives the demo workflow.

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/api/v1/config/ports` | Departure ports |
| GET | `/api/v1/config/research-centers` | Research-center destinations |
| GET | `/api/v1/config/simulation` | Simulation defaults for the frontend |
| GET | `/api/v1/vessels/`, `/api/v1/vessels/{id}` | Vessel catalog |
| POST | `/api/v1/navigation/journey` | Start journey (outbound/return, sim/live) |
| POST | `/api/v1/navigation/journey/{id}/advance` | Advance 2 h + recalculate from current position |
| POST | `/api/v1/navigation/journey/{id}/recalculate?lat=&lon=` | Recalculate from live fix |
| GET | `/api/v1/navigation/journey/{id}/status` | Position, progress, time, fuel, completion |
| GET | `/api/v1/sea-ice/forecast?horizon_hours=&model=` | Sea-ice forecast |
| GET | `/api/v1/sea-ice/models` | Sea-ice models |
| GET | `/api/v1/icebergs/predictions?horizon_hours=` | Iceberg trajectory predictions |
| GET | `/api/v1/icebergs/forecast-grid` | Predictions at all horizons |
| GET | `/api/v1/icebergs/models` | Iceberg models |
| POST | `/api/v1/routes/optimize` | Decisions-support route optimization (recommended + alternatives) |

### Journey lifecycle example

```bash
# 1. start an outbound journey Hobart → Casey (simulation mode)
curl -X POST http://localhost:8001/api/v1/navigation/journey \
  -H "Content-Type: application/json" \
  -d '{"departure_port_id":"hobart_au","destination_id":"casey_au",
       "journey_mode":"outbound","position_mode":"simulation"}'

# 2. advance two simulated hours (route recalculates from current position)
curl -X POST http://localhost:8001/api/v1/navigation/journey/<id>/advance

# 3. recalculate from an external (GPS) fix
curl -X POST "http://localhost:8001/api/v1/navigation/journey/<id>/recalculate?lat=-68.0&lon=120.0"

# 4. status
curl http://localhost:8001/api/v1/navigation/journey/<id>/status
```

Live mode (`position_mode: "live"`) returns 400 until a real position is
supplied via `/recalculate`.

## Frontend usage

- `frontend/src/services/api.ts` — typed axios client.
- Pages hit the `/api/*` surface; the demo workflow additionally drives the
  legacy journey API from `backend/scripts/demo_workflow.py`.