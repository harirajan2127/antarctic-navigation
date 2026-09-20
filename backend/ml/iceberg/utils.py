"""Shared helpers for the iceberg trajectory forecasting package.

Kept dependency-light (numpy + stdlib) so trajectory math can be unit-tested
in isolation. All file I/O is UTF-8 for cross-platform safety.
"""
from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

LOG = logging.getLogger(__name__)

# Mean Earth radius (km), IUGG standard value.
EARTH_RADIUS_KM = 6371.0088
NAUTICAL_MILES_PER_KM = 1.0 / 1.852

DISCLAIMER_DEMO = (
    "DEMO MODE: trained/validated on SYNTHETIC demo data. No real observations "
    "were used. All numbers in this run are code-path checks and carry NO "
    "information about real-world forecasting skill."
)
DISCLAIMER_TRAJECTORY = (
    "Predicted trajectories are statistical extrapolations of an ML model, NOT "
    "guaranteed positions. Always treat them as advisory and re-validate."
)


def set_seed(seed: int) -> None:
    """Reproducibility for numpy, standard library and torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def configured_logger(verbose: bool = False) -> logging.Logger:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    return root


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Geodesy (self-contained spherical math)
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    r1 = np.deg2rad(lat1)
    r2 = np.deg2rad(lat2)
    dr = np.deg2rad(lat2 - lat1)
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dr / 2) ** 2 + np.cos(r1) * np.cos(r2) * np.sin(dl / 2) ** 2
    return float(2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def haversine_degrees_km(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Pairwise distance between consecutive (lat, lon) rows -> km array."""
    out = np.zeros(len(lat_deg) - 1)
    for i in range(len(out)):
        out[i] = haversine_km(lat_deg[i], lon_deg[i], lat_deg[i + 1], lon_deg[i + 1])
    return out


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2 (degrees 0..360)."""
    r1, r2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dl = np.deg2rad(lon2 - lon1)
    y = np.sin(dl) * np.cos(r2)
    x = np.cos(r1) * np.sin(r2) - np.sin(r1) * np.cos(r2) * np.cos(dl)
    return float((np.rad2deg(np.arctan2(y, x)) + 360.0) % 360.0)


def wrap_lon(lon) -> np.ndarray | float:
    """Wrap longitude(s) into the [-180, 180) range (survives equirect planes)."""
    arr = np.asarray(lon, dtype=np.float64)
    wrapped = (arr + 180.0) % 360.0 - 180.0
    if np.ndim(arr) == 0:
        return float(wrapped)
    return wrapped


# ---------------------------------------------------------------------------
# Local projection
# ---------------------------------------------------------------------------

class LocalTangentPlane:
    """Equirectangular (plate carree) tangent-plane projection.

    Centres the axes on an origin (lat0, lon0) and returns X (east), Y (north)
    in **kilometres**. Spherical with local radius ``cos(lat0)`` scaling for X.
    Distortion is negligible for the track lengths icebergs traverse in a
    forecasting horizon (< ~0.5% over 100 km), and the inverse transform is
    exact so lat/lon round-trips are lossless.
    """

    def __init__(self, lat0: float, lon0: float):
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        self.km_per_deg = np.pi * EARTH_RADIUS_KM / 180.0
        self.km_per_deg_x = self.km_per_deg * np.cos(np.deg2rad(lat0))
        self.units = "km"

    def forward(self, lat: np.ndarray | float, lon: np.ndarray | float) -> tuple:
        lon = np.asarray(lon, dtype=np.float64)
        x = (lon - self.lon0) * self.km_per_deg_x  # km (eastward)
        y = (np.asarray(lat, dtype=np.float64) - self.lat0) * self.km_per_deg
        return x, y

    def inverse(self, x, y) -> tuple:
        lon = self.lon0 + np.asarray(x) / self.km_per_deg_x
        lat = self.lat0 + np.asarray(y) / self.km_per_deg
        return lat, lon

    @classmethod
    def origin_of(cls, lat: np.ndarray, lon: np.ndarray) -> "LocalTangentPlane":
        return cls(float(lat[0]), float(lon[0]))


# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------

def sin_cos_doy(times) -> tuple[np.ndarray, np.ndarray]:
    """Cyclic day-of-year features in [0, 1] from any datetime-like array."""
    dts = np.asarray(times, dtype="datetime64[us]").astype("datetime64[D]")
    doy = (dts.astype("int64") - dts.astype("datetime64[Y]").astype("int64")) + 1
    ang = 2.0 * np.pi * (doy - 1.0) / 366.0
    return (np.sin(ang) + 1.0) / 2.0, (np.cos(ang) + 1.0) / 2.0


def is_demo_dataset(paths: Iterable[Path | str]) -> bool:
    """True if any NetCDF env file (or demo CSV) signals synthetic data."""
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        if p.suffix == ".nc" and p.exists():
            import xarray as xr

            with xr.open_dataset(p) as ds:
                cls = str(ds.attrs.get("classification", "")).lower()
            if "demo" in cls or "synthetic" in cls:
                return True
        elif p.suffix == ".csv":
            if "demo" in p.stem.lower():
                return True
            # A CSV can carry demo icebergs without a demo filename.
            try:
                import pandas as pd

                head = pd.read_csv(p, usecols=["iceberg_id"], nrows=20)
                if any("demo" in str(i).lower() for i in head["iceberg_id"]):
                    return True
            except Exception:
                pass
    return False


def explain_data_situation(demo_data: bool, demo_flag: bool) -> None:
    if demo_data and not demo_flag:
        raise SystemExit(
            "\n[iceberg] The only available tracks/env data is SYNTHETIC (demo).\n"
            "  A real trajectory model CANNOT be meaningfully trained on it.\n"
            "  Options:\n"
            "    1. Provide real CSV tracks + NetCDF ocean/weather files.\n"
            "    2. Run the *demo smoke test* with  --demo  (exercises the full\n"
            "       code path, produces no meaningful skill numbers).\n"
        )
    if demo_data and demo_flag:
        LOG.warning("Running in DEMO mode on synthetic data. Metrics are NOT meaningful.")


def save_json(obj: Any, path: Path | str, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, default=_json_default, ensure_ascii=False)
    return path


def load_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_md(text: str, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _json_default(o: Any):
    if isinstance(o, (np.integer, np.floating)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (datetime, np.datetime64)):
        return str(o)
    return str(o)