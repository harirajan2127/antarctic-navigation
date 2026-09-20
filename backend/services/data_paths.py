"""Resolved paths to datasets.

Constants here are the single point where the loaders learn where the *real*
datasets live. They are read in this order of preference:

1. ``Real data/processed/<dataset>/csv/`` — the real, preprocessed dataset.
2. ``Real data/raw/<dataset>/`` — the raw original (used only if the processed
   copy is missing).
3. A missing sentinel path — when the app runs in real or auto mode without the
   expected real file, the API reports *Data Unavailable* instead of silently
   switching to synthetic demo data.

The demo files under ``backend/datasets/processed/`` remain on disk for tests and
offline pipeline generation, but they are never used as a runtime fallback for the
main application when the user is expecting real data.
"""
from __future__ import annotations

import warnings
from pathlib import Path

from config import settings

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BACKEND_DIR / "datasets" / "processed"

_REAL_ROOT = BACKEND_DIR.parents[0] / "Real data"

_RAW = _REAL_ROOT / "raw"
_PROC = _REAL_ROOT / "processed"
_CSV = {ds: _PROC / ds / "csv" for ds in
        ("sea_ice", "iceberg", "ocean", "weather", "vessel", "bathymetry")}

_DEMO_SEA_ICE = PROCESSED_DIR / "sea_ice.nc"
_DEMO_OCEAN = PROCESSED_DIR / "ocean_surface.nc"
_DEMO_WEATHER = PROCESSED_DIR / "weather_surface.nc"
_DEMO_ICEBERG_CSV = PROCESSED_DIR / "icebergs_demo.csv"


def _first_csv(folder: Path) -> Path | None:
    """Return the first (sorted) CSV in a processed dataset csv folder, if any."""
    if not folder.is_dir():
        return None
    files = sorted(p for p in folder.glob("*.csv") if p.is_file())
    return files[0] if files else None


def _resolve(real_candidates: list[Path], demo: Path, label: str) -> Path:
    for path in real_candidates:
        if path.is_file():
            return path

    mode = settings.DATA_MODE.strip().lower()
    if mode == "demo":
        return demo

    # Real/offline-safe mode: never silently switch to the synthetic demo files.
    warnings.warn(
        f"Real dataset '{label}' not found under Real data/raw|processed or the "
        f"configured real path while DATA_MODE={mode}. The API will mark this "
        "dataset unavailable instead of using synthetic demo data.",
        UserWarning,
        stacklevel=3,
    )
    return _PROC / label / "csv" / "__missing__"


def _has_real_data() -> bool:
    return _REAL_ROOT.exists() and (
        (_RAW.is_dir() and any(_RAW.iterdir()))
        or (_PROC.is_dir() and any(_PROC.iterdir()))
    )


# Primary platform datasets. Each resolves to a real CSV when one exists.
_REAL_SEA_ICE_CSV = _first_csv(_CSV["sea_ice"])
_REAL_OCEAN_CSV = _first_csv(_CSV["ocean"])
_REAL_WEATHER_CSV = _first_csv(_CSV["weather"])
_REAL_ICEBERG_CSV = _first_csv(_CSV["iceberg"])

SEA_ICE_NETCDF = _resolve(
    [_REAL_SEA_ICE_CSV] if _REAL_SEA_ICE_CSV else [_PROC / "sea_ice" / "csv"],
    _DEMO_SEA_ICE,
    "sea_ice",
)
OCEAN_NETCDF = _resolve(
    [_REAL_OCEAN_CSV] if _REAL_OCEAN_CSV else [_PROC / "ocean" / "csv"],
    _DEMO_OCEAN,
    "ocean",
)
WEATHER_NETCDF = _resolve(
    [_REAL_WEATHER_CSV] if _REAL_WEATHER_CSV else [_PROC / "weather" / "csv"],
    _DEMO_WEATHER,
    "weather",
)

_DEMO_ICEBERG_CSV = PROCESSED_DIR / "icebergs_demo.csv"
ICEBERG_CSV = _REAL_ICEBERG_CSV if (_REAL_ICEBERG_CSV := _first_csv(_CSV["iceberg"])) else (
    _resolve([_PROC / "iceberg" / "csv"], _DEMO_ICEBERG_CSV, "iceberg")
)

have_real_datasets: bool = _has_real_data() and (
    _first_csv(_CSV["sea_ice"]) is not None or _first_csv(_CSV["iceberg"]) is not None
)
