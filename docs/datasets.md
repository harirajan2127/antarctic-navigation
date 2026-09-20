# Datasets

All datasets are processed into `backend/datasets/processed/` and consumed
directly by the running services. When no upstream credentials are configured,
the pipeline generates **clearly-labeled synthetic demo data** — it is never
presented as real.

## Source registry

| Domain | Real sources (when configured) | Current status |
|---|---|---|
| Sea ice concentration | NSIDC Sea Ice Index, Copernicus Marine | demo (synthetic) |
| Iceberg tracks | NSIDC G00803, US National Ice Center | demo (synthetic) |
| Ocean surface fields | NASA OSCAR, Copernicus Marine | demo (synthetic) |
| Weather | ECMWF ERA5 | demo (synthetic) |
| Bathymetry / land | land polygon mask (numpy `.npy`) | real mask shipped |

Source metadata lives in `backend/app/data/config/datasets.json`.

## Layout

```
backend/datasets/
├── raw/                  # provider downloads (SHA-256 verified, skip-existing)
├── processed/            # normalized inputs used by the app:
│   ├── sea_ice.nc        #   (time, latitude, longitude) concentration
│   ├── ocean_surface.nc  #   uo, vo, thetao, so
│   ├── weather_surface.nc#   u10, v10, t2m, msl
│   ├── icebergs.csv / .nc#   imported tracks (real) 
│   └── icebergs_demo.*   #   synthetic demo tracks
├── imports/              # source copies of imported iceberg files
└── reports/              # dataset_report.json / .md
```

The service layer prefers the **real** CSV (`icebergs.csv`) when present; only
when it is absent does a demo build fall back to `icebergs_demo.*`.

## Classification & honesty

- Every artifact carries `classification: synthetic_demo` or `real`.
- A `real`/`auto` run never silently reuses an existing `synthetic_demo` file —
  it logs a warning and regenerates from the real source.
- Existing valid files are skipped by default (`PIPELINE_SKIP_EXISTING`);
  `--force` deletes before re-acquiring.
- `GET /api/datasets/status` exposes per-dataset status and the aggregate
  `real_data_available` flag (true only when real — not demo — data is
  serving a given domain). The flag also appears on `GET /api/analytics/summary`.

## Generation / download commands (run from `backend/`)

```bash
# Offline synthetic demo bundle (no credentials required)
python scripts/create_demo_data.py --days 30

# Provider downloads — demo mode (synthetic) or auto (real when creds exist)
python scripts/download_sea_ice.py --mode demo --days 30
python scripts/download_ocean.py   --mode demo --days 30
python scripts/download_weather.py --mode demo --days 30

# Iceberg import: real CSV or synthetic demo track
python scripts/import_icebergs.py path/to/icebergs.csv --tag icebergs_survey
python scripts/import_icebergs.py --demo
```

## Preprocessing & validation

```bash
python data/run_preprocessing.py       # raw → processed (real CSV preferred)
python scripts/validate_datasets.py    # → datasets/reports/dataset_report.{json,md}
```

Validation checks variable presence, units, coordinate/date ranges, missing
values, cadence and freshness (age of the newest observation).

## Cadence vs the 2-hour loop

Remote observations are daily (sea-ice/ocean) or 6-hourly (ERA5), but the
navigation loop runs every 2 simulated hours. `data_pipeline/processor/accessor.py`
serves the closest valid observation with **provenance metadata** — original
timestamp, method (`nearest`/linear), freshness report. It never claims a
daily field is 2-hourly.