# Demo

The demo runs entirely offline with **clearly-labeled synthetic data**
(`classification: synthetic_demo`). None of it is presented as real.

- 6 departure ports: Hobart, Christchurch, Bluff, Cape Town, Ushuaia,
  Punta Arenas
- 15+ configured Antarctic research centers (coordinates verified at config
  time; `sample_configuration: true`)
- Vessel catalog with ice class, speed and fuel model
- Sea-ice, iceberg, ocean and weather demo fields (30 days generated)

## One-command bootstrap (Windows)

```powershell
powershell -ExecutionPolicy Bypass -File run_demo.ps1
```

`run_demo.ps1` performs the full accepted workflow:

1. ensures processed demo datasets (`create_demo_data.py --days 30`);
2. starts the **unified backend** (`uvicorn main:app` on :8000);
3. starts the **legacy journey API** (`uvicorn app.main:app` on :8001);
4. starts the **frontend** (`npm run dev` on :5173);
5. waits for both health checks;
6. runs the demo-integrity check (`scripts/check_demo.py`);
7. drives the end-to-end workflow (`scripts/demo_workflow.py --steps 2
   --return`).

Flags: `-SkipInstall` (fast path), `-SkipDemo`.

## The 10-step demo workflow

Performed automatically by `demo_workflow.py` against the live legacy API
(`http://127.0.0.1:8001`), or manually in the browser / Swagger UI:

| # | Step | Where |
|---|---|---|
| 1 | Start backend | `run_demo.ps1` starts :8000 + :8001 |
| 2 | Start frontend dashboard | `npm run dev` → http://localhost:5173 |
| 3 | Load/verify demo data | `/api/datasets/status` (`real_data_available=false`, datasets demo) |
| 4 | Select departure port | `POST /api/v1/navigation/journey` `departure_port_id` (e.g. `hobart_au`) |
| 5 | Select research center | same payload `destination_id` (e.g. `casey_au`) |
| 6 | Start journey (outbound) | response carries route, distance, fuel, risk |
| 7 | Advance 2 h | `POST .../journey/{id}/advance` — position moves, `route_update_count` grows |
| 8 | Recalculate route | `POST .../journey/{id}/recalculate?lat=&lon=` (or auto-recalc on advance from current position) |
| 9 | Show map + metrics | frontend Dashboard (Leaflet map, distance/fuel/risk cards) |
| 10 | Return journey | start a second journey with `"journey_mode":"return"` |

Manual equivalent:

```powershell
cd backend
python scripts/demo_workflow.py --steps 2 --return          # Hobart → Casey
python scripts/demo_workflow.py --port ushuaia_ar --center mcmurdo_us --steps 2
python scripts/demo_workflow.py --steps 0 --return          # advance until arrival
```

## Scripts

| Script | Purpose |
|---|---|
| `backend/scripts/create_demo_data.py` | generate offline demo datasets (`--days`) |
| `backend/scripts/check_demo.py` | demo-integrity checks against the running backend |
| `backend/scripts/demo_workflow.py` | drives the 10-step workflow over HTTP (stdlib only) |
| `backend/scripts/validate_datasets.py` | dataset report / freshness |

## Health endpoints

- `http://127.0.0.1:8000/api/health`
- `http://127.0.0.1:8001/health`
- Dashboard: `http://localhost:5173`

Everything demo-labeled: dataset status shows `synthetic_demo`, models report
`trained_on_demo: true`, and the UI keeps a persistent Demo Mode banner with a
route-safety disclaimer.