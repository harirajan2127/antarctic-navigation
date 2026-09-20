# Architecture

The DSS is a three-tier web application: a React frontend, a FastAPI backend
(split into two HTTP applications), and a local data/ML layer. Everything runs
locally on one machine for the current phase.

```
┌──────────────────────────────────────────────────────────────────┐
│ React + Vite frontend (TypeScript)                                │
│ Tailwind · Leaflet (AntarcticMap) · Recharts · axios              │
│ http://localhost:5173  —  Vite dev proxy forwards /api → :8000    │
└───────────────┬──────────────────────────────┬────────────────────┘
                │ /api/* (JSON)                │ (uses legacy data too)
┌───────────────▼──────────────────────────────▼────────────────────┐
│ FastAPI application #1 — unified REST API                          │
│   uvicorn main:app  ·  http://localhost:8000  ·  prefix /api       │
│   routers: health, datasets, sea-ice, icebergs, routes, vessels,   │
│            analytics, models/status, assistant/chat                │
└───────────────┬────────────────────────────────────────────────────┘
┌───────────────▼────────────────────────────────────────────────────┐
│ FastAPI application #2 — legacy journey lifecycle                  │
│   uvicorn app.main:app  ·  http://localhost:8001  ·  prefix /api/v1 │
│   routers: config, vessels, navigation (journey/advance/recalc),   │
│            sea-ice, icebergs, routes/optimize                      │
└───────────────┬────────────────────────────────────────────────────┘
                │ same-process import of services
┌───────────────▼────────────────────────────────────────────────────┐
│ Backend core (shared by both apps)                                 │
│   navigation/    numpy-only engine: grid, risk_engine, astar,      │
│                  route_optimizer, fuel_estimator, geodesy,         │
│                  land_mask (Antarctic land .npy)                    │
│   services/      sea_ice_service, iceberg_service, navigation_service,
│                  dataset_service, analytics_service, provenance,   │
│                  distance_calculator, model_registry               │
│   ml/            sea_ice (persistence, RF, ConvLSTM)               │
│                  iceberg (persistence, drift, RF, LSTM)            │
│   api/           legacy route modules for /api/v1                  │
│   assistant/     tool-backed chat (fallback prompt when no LLM)    │
│   database/      SQLAlchemy + SQLite (inventory, analytics, routes)│
│   data_pipeline/ acquisition → validation → report                 │
└───────────────┬────────────────────────────────────────────────────┘
                │ filesystem artifacts
┌───────────────▼────────────────────────────────────────────────────┐
│ Data & models                                                     │
│   backend/datasets/raw|processed|imports|reports                  │
│   backend/models/sea_ice/run  ·  backend/models/iceberg/run       │
│   backend/app/data/config/*.json                                  │
│   backend/data/config/ports.json, research_centers.json, ...      │
└────────────────────────────────────────────────────────────────────┘
```

## Components

### Frontend (`frontend/`)
Pages: Overview, Sea-Ice Forecast, Iceberg Tracking, Navigation Planner,
Analytics, AI Assistant. The map component
(`frontend/src/components/map/AntarcticMap.tsx`) renders sea-ice grids
(canvas overlay), iceberg markers, and route polylines on Leaflet.

### Unified REST API (`backend/main.py`)
The primary surface for the dashboard. All responses label demo data
(`demo=true`, `warning`), report `model_used_real`, and the analytics summary
carries a `real_data_available` flag. See `docs/api.md`.

### Legacy journey API (`backend/app/main.py`)
Owns the **journey lifecycle** used by the demo workflow:
start journey (`POST /api/v1/navigation/journey`), advance two simulated
hours (`POST .../advance`), recalculate from a live position
(`POST .../recalculate?lat=&lon=`), status (`GET .../status`). Modeled as
in-memory journeys managed by `app/services/simulation/simulator.py`.

### Navigation engine (`backend/navigation/`)
Pure-numpy decision support engine. Grid domain: latitude −85°…−55°,
longitude −180°…180°, 0.5° resolution. Risk = weighted sea-ice risk + hard
iceberg No-Go blocks + weather + vessel ice-class constraints.
A\* path planning over the risk-weighted cost field, then a fuel/time
estimator. See `docs/navigation.md`.

### ML pipelines (`backend/ml/`)
Sea-ice forecasting (persistence / Random Forest / ConvLSTM) and iceberg
trajectory prediction (persistence / current-drift / Random Forest / LSTM).
Trained artifacts are loaded at runtime through `ml/*/runtime.py` and
advertised via `GET /api/models/status`. See `docs/model_training.md`.

### Data pipeline (`backend/data_pipeline/`)
Acquisition, validation and reporting. Demo workflows generate explicitly
labeled synthetic data offline; real sources (Copernicus, NSIDC, ERA5, OSCAR)
are used only when `.env` credentials exist. See `docs/datasets.md`.

### Storage
- SQLite via SQLAlchemy (`backend/database/`): dataset inventory, analytics
  ingestion/latency metrics, optimizer route snapshots.
- Filesystem: processed NetCDF/CSV, trained artifacts, reports.

## Two-hour rolling loop

At each `SIMULATION_TIME_STEP_HOURS = 2` hour boundary the journey manager:
1. advances the vessel along the current route,
2. reloads the closest valid observation set with provenance,
3. rebuilds the risk grid,
4. re-plans the route **from the current position** (never resetting to the
   departure port), and
5. recomputes distance / fuel / duration / risk + a route-update counter.

## Honesty model
- Demo artifacts are stamped `classification: synthetic_demo`; demo-trained
  models report `trained_on_demo: true`.
- No real model metrics are invented (untrained → `Not trained` + fallback).
- Routes are decision support only, never "guaranteed safe".
- No live GPS is claimed unless a position is supplied via the recalc API.