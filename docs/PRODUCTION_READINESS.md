# Production-Readiness Checklist

This document summarises the hardening, correctness fixes, and verification
performed in the final release phase of the Antarctic Navigation DSS. It is the
paper trail for *what was changed and why*, so every claim the system makes can
be audited.

## 1. Honest-data policy (enforced end to end)

| Claim | Enforcement |
|---|---|
| Synthetic data is never presented as real | `data_pipeline/common/netcdf_utils.py` stamps `classification: synthetic_demo`; every API response carries `demo` / `classification` / `warning`. |
| A demo-classified file never satisfies a "real" download | `data_pipeline/sea_ice/acquire.py::ensure_sea_ice` checks the existing file's classification and regenerates from the real source when `mode=real`/`auto` (instead of silently returning the demo file). |
| Real artifacts are preferred over demo ones | Preprocessing (`data/run_preprocessing.py::_default_paths`) and validation (`data_pipeline/icebergs/import_data.py::validate_processed`) select `icebergs.csv` before `icebergs_demo.csv`. |
| `real_data_available` reflects any non-demo dataset | `services/dataset_service.py::real_data_available` treats *any* non-demo classification (e.g. `real_observation`) as real, not just `pipeline_data`. Surfaced in `/api/datasets/status` **and** `/api/analytics/summary`. |
| Models trained on demo data are clearly labelled | Registry reports `trained_on_demo` per checkpoint; runtimes report `model_used_real: false`; UI shows an amber `demo` badge (Analytics → Trained Models). |
| Unreliable models fall back, never crash | `SeaIceRuntime` / `IcebergRuntime` set `available: false` + `error` on any load failure; services fall back to persistence. `grid_predict()` raises `RuntimeError` so callers cannot silently misuse it. |
| Land-mask absence is explicit | `navigation/land_mask.py` returns `(None, info)` when `LAND_MASK_FILE` is unset/missing/unreadable and the API surfaces it as a route note ("open water assumed"), never silently. |

## 2. Data-pipeline correctness fixes

1. **Forced refresh** — `RobustDownloader.download(..., force=True)` deletes the
   existing file and re-downloads; the "already exists and is valid" skip
   (`ExistingFileSkipped`) is bypassed only under `force`. (Also fixed: the skip
   fired on *valid* files even under `force`.)
2. **Demo-blocking for real runs** — an existing `sea_ice.nc` classified
   `synthetic_demo` no longer satisfies `mode=real` or `mode=auto` with
   credentials configured.
3. **Real-CSV preference** — `run_preprocessing.py` and
   `icebergs/import_data.py::validate_processed` prefer `icebergs.csv`.
4. **`real_data_available` widening** — any non-demo classification counts, and
   the flag is now also available on the analytics summary.
5. **Analytics seeding bug (count 0)** — root cause: the SQLAlchemy session is
   `autoflush=False` (`database/database.py`), so the clear (DELETE) in
   `seed_model_metrics_from_artifacts` was never flushed and the subsequent
   INSERT lookups still saw old rows, taking an UPDATE path that silently
   wiped everything at commit. Fix: `db.flush()` after `_clear_metrics()` plus a
   tz-naivety guard in `_insert_metric`. Seeding is verified idempotent
   (pre:31 → post:31; pre:0 → post:31).

## 3. Model registry & live runtimes

- `services/model_registry.py` — read-only aggregation of
  `backend/models/{sea_ice,iceberg}/run`, exposed at `GET /api/models/status`
  (`models`, `count`, `any_real_trained`, `config`, `demo_mode`, `warning`).
- `ml/sea_ice/runtime.py` / `ml/iceberg/runtime.py` — in-memory inference using
  the same code path as the training CLIs; cached per (kind, out_dir).

## 4. Navigation hardening

- **Iceberg hard block (No-Go)** — `RiskConfig.hard_iceberg_block_km`
  (20.0 km in `navigation.json`); `RiskEngine.apply_iceberg_hard_block()`
  forces cells inside the radius into `blocked_mask` (total risk 1.0), so A*
  treats them as obstacles rather than high-cost cells.
- **Land-mask loader** — NetCDF (elevation/depth), aligned `.npy`, and GeoTIFF
  inputs mapped onto `AntarcticGrid`; pre-aligned `.npy` masks are returned
  directly. Fixed a latent `UnboundLocalError` on the npy path.
- `RouteValidator` + `is_navigable` checks reject routes crossing land /
  out-of-bounds (existing tests extended).

## 5. Tests (backend: 203 passed)

- `tests/test_data_pipeline_behavior.py` — downloader skip vs. force,
  demo-blocking, real-CSV preference and fallback.
- `tests/test_ml_api_integration.py` — registry (empty/demo/real/corrupt-json),
  sea-ice & iceberg runtime error paths and `step_hours_for` NaN guard, iceberg
  hard-block unit + risk interplay, land-mask loading (none/missing/npy/netcdf,
  integration into RiskEngine).
- `tests/test_api_system.py` — analytics seeding now stable in the full suite.

Run everything: `cd backend; python -m pytest tests -q`.

## 6. Frontend

- Analytics page gained a **Trained Models** card (`GET /api/models/status`)
  showing availability, demo/real badge, artifact, and per-model errors.
- `npm test` = `tsc --noEmit && vite build` (no test-runner dependency);
  verified a clean production build.