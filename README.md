# Antarctic Navigation DSS

This project is an Antarctic decision-support system for sea-ice monitoring, iceberg forecasting, and vessel route planning. The current application is a full-stack system with a unified FastAPI backend and a React + Vite frontend.

## Current process in this repository

The application is currently run in this order:

1. The root launcher starts the backend and frontend together.
2. The backend serves the REST API from `backend/main.py` on port `8000`.
3. The frontend runs from `frontend/` on port `5173` and calls the backend through the Vite proxy.
4. The app loads data from either the synthetic demo datasets or the real processed datasets depending on configuration.
5. Navigation routes are optimized through the backend route engine and returned to the frontend for display.

This is the live setup used by the repository today.

## Architecture

- Frontend: React + TypeScript + Vite + Tailwind + Leaflet
- Backend: FastAPI with SQLAlchemy and a unified `/api` surface
- Data layer: processed datasets, route engine, model loaders, and validation scripts
- Runtime ports:
  - Backend: http://localhost:8000
  - Frontend: http://localhost:5173

## Start the app

From the project root:

```powershell
.\run.ps1
```

This script:

- creates or refreshes the Python virtual environment in `.venv`
- installs backend dependencies from `backend/requirements.txt` if needed
- installs frontend dependencies from `frontend/package.json` if needed
- starts the backend with:

```powershell
python -m uvicorn main:app --reload --port 8000
```

- starts the frontend with:

```powershell
npm run dev
```

Use the skip-install path after the first setup:

```powershell
.\run.ps1 -SkipInstall
```

## Project layout

```text
SIH HACK/
├── backend/
│   ├── api/                  # FastAPI route modules
│   ├── app/                 # legacy app layer + route modules
│   ├── database/            # SQLAlchemy DB setup and models
│   ├── main.py              # unified backend entry point
│   ├── config.py            # settings, env variables, demo/real mode
│   ├── services/            # route, dataset, model, and analytics services
│   ├── ml/                  # sea-ice and iceberg model code
│   ├── tests/               # backend test suite
│   ├── requirements.txt
│   └── ...
├── frontend/
│   ├── src/                 # dashboard pages, map, types, API client
│   ├── package.json
│   ├── vite.config.ts
│   └── ...
├── scripts/
│   ├── preprocess_real_data.py
│   ├── run_real_pipeline.py
│   ├── evaluate_real_models.py
│   ├── calculate_total_accuracy.py
│   └── ...
├── Real data/
│   ├── processed/
│   └── README.md
├── reports/
├── docs/
├── .env.example
├── run.ps1
├── run_accuracy.ps1
├── README.md
└── ...
```

## Backend entry point

The active backend entry point is:

- `backend/main.py`

It mounts the unified API under `/api` and keeps the legacy `/api/v1` routes for compatibility. The app initializes the database and seeds resources on startup.

Key routes include:

- `/api/health`
- `/api/datasets`
- `/api/sea-ice/current`
- `/api/sea-ice/forecast`
- `/api/icebergs`
- `/api/routes/optimize`
- `/api/assistant/chat`
- `/api/analytics/summary`

API docs are available at:

- http://localhost:8000/docs
- http://localhost:8000/openapi.json

## Frontend app

The dashboard is served from `frontend/` and uses Vite dev mode by default. The app consumes the backend `/api` routes and renders:

- sea-ice forecast pages and overlays
- iceberg tracking and trajectory views
- navigation planner with optimized routes
- analytics dashboards
- AI assistant panel

## Data flow and mode selection

The app supports both demo and real-data modes through environment settings in `backend/config.py`:

- `DATA_MODE` controls whether the app prefers real or synthetic data
- `DEMO_MODE` controls how demo labeling is reported
- The default configuration is designed to work with the repository data and can fall back to synthetic demo data where needed

The data processing workflow under `scripts/` and `Real data/` is used for real-data preparation and validation. The app is expected to read processed outputs before serving live endpoints.

## Current workflow process

The actual project workflow today is:

1. Ensure environment is configured from `.env.example`.
2. Run the root launcher: `./run.ps1`.
3. Backend starts on port 8000 and initializes DB + seed data.
4. Frontend starts on port 5173 and calls the backend through Vite.
5. User interacts with the dashboard to view sea-ice, iceberg movement, and route optimization.
6. Route optimization and analytics are generated from the backend services, not from hard-coded frontend values.
7. Validation and performance checks can be run through the scripts in `scripts/`.

## Common commands

Backend:

```powershell
cd backend
python -m uvicorn main:app --reload --port 8000
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

Real-data evaluation:

```powershell
.\run_accuracy.ps1
```

This script runs the accuracy calculation based on the project’s evaluation/reporting pipeline.

## Safety and disclaimers

This system is a decision-support tool, not a guaranteed-safe navigation system. Routes, forecasts, and predictions should be treated as operational guidance only. Demo data is clearly labeled when used, and the backend is designed to avoid presenting synthetic outputs as real-world information.

## Notes

The repository contains both a unified current API and a legacy app layer. The active application is the unified FastAPI app in `backend/main.py`, and the root launcher reflects that current process.
