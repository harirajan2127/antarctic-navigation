# Real data

This folder is the single home for the **real (non-demo)** datasets used by the
Antarctic Navigation DSS. Everything under `raw/` is **your original data,
exactly as you placed it** — this project never modifies, cleans, resamples or
overwrites it. Everything under `processed/` holds files this project
generates from your raw data (it never touches the originals). `external/` is
for third-party reference files (coastlines, land polygons, bathymetry) that
help the validators classify reports.

> **The dashboard keeps working while these folders are empty.** When a
> dataset is missing the app falls back to the demo/synthetic-demo datasets
> and shows a clear "real dataset missing" warning — it never crashes.

## Where to place each dataset

| Dataset id | Raw folder (drop originals here)      | Processed folder (generated)          |
|------------|---------------------------------------|---------------------------------------|
| sea_ice    | `Real data/raw/sea_ice/`              | `Real data/processed/sea_ice/`        |
| iceberg    | `Real data/raw/iceberg/`              | `Real data/processed/iceberg/`        |
| ocean      | `Real data/raw/ocean/`                | `Real data/processed/ocean/`          |
| weather    | `Real data/raw/weather/`              | `Real data/processed/weather/`        |
| vessel     | `Real data/raw/vessel/`               | `Real data/processed/vessel/`         |

External/reference files go in `Real data/external/`.

## Supported file formats

- **Tabular:** CSV, JSON, GeoJSON, Parquet, GeoPackage (`.gpkg`)
- **Gridded / raster:** NetCDF (`.nc`, `.netcdf`), TIFF / GeoTIFF (`.tif`, `.tiff`)

## Expected variables / columns

| Dataset | Required variables/columns             | Conditions                                    |
|---------|----------------------------------------|-----------------------------------------------|
| sea_ice | `concentration`, `latitude`, `longitude` | concentration typically 0–1 (fraction)        |
| iceberg | `latitude`, `longitude` (+ `on_land`)   | rows at sea; `on_land` flags over-continent ones |
| ocean   | `sst`, `latitude`, `longitude`          | Antarctic region                              |
| weather | `wind_u`, `wind_v` (+ `pressure`, `temperature`) | surface fields                         |
| vessel  | `latitude`, `longitude` (+ `speed`, `heading`) | vessel state                           |

## How to run dataset validation

Requires Python with PyYAML (installed in this project).

```bash
cd "C:\my project\SIH HACK"
python scripts/validate_datasets.py
```

Exit code `0` = all checks passed (warnings allowed); `1` = at least one
dataset reported an error. The checker is read-only — it never writes.

## How to start preprocessing

Preprocessing outputs are written **only** under `Real data/processed/…` and
never touch your original files:

```bash
cd "C:\my project\SIH HACK"
pip install -r backend/requirements.txt
python backend/data/run_preprocessing.py
```

Loaders in the backend resolve datasets from these folders first; if a folder
is empty they fall back to demo data with a warning.
