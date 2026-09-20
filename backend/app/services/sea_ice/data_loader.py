"""Sea-ice data loading from pipeline output, CSV, NetCDF, or synthetic samples.

Priority:
1. Real processed CSV (``Real data/processed/sea_ice/csv/*.csv``) — the newest
   time step is pivoted into the concentration grid the API serves.
2. Pipeline-processed file (``backend/datasets/processed/sea_ice.nc``) written by
   ``scripts/download_sea_ice.py`` / ``scripts/create_demo_data.py``.
3. An explicitly-configured NetCDF path.
4. Clearly-labeled synthetic sample data.

When ``DATA_MODE=real`` and no real CSV exists the loader raises no exception
but returns an empty grid tagged ``classification="unavailable"`` so the API
surfaces *Data Unavailable* instead of synthetic demo data.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.utils.constants import ANTARCTIC_LAT_MAX, ANTARCTIC_LAT_MIN, ANTARCTIC_LON_MAX, ANTARCTIC_LON_MIN

from config import settings

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "data" / "sample"
PIPELINE_PROCESSED = Path(__file__).resolve().parents[2] / "datasets" / "processed" / "sea_ice.nc"
REAL_CSV_DIR = Path(__file__).resolve().parents[4] / "Real data" / "processed" / "sea_ice" / "csv"


class SeaIceDataError(Exception):
    """Raised when sea-ice data cannot be loaded."""


class SeaIceDataLoader:
    """Loads sea-ice concentration data from the pipeline or synthetic samples."""

    def __init__(self, netcdf_path: str | None = None) -> None:
        self.netcdf_path = Path(netcdf_path) if netcdf_path else None

    def load(self, timestamp: datetime | None = None) -> dict[str, Any]:
        """Load sea-ice concentration data.

        Prefers the real processed CSV (newest time step), then the
        pipeline-processed dataset, then an explicitly configured NetCDF.
        Returns data tagged with provenance so callers can distinguish demo
        from real sources. When ``DATA_MODE=real`` and no real data exists the
        result is tagged ``unavailable`` (never synthetic).
        """
        csv_path = self._real_csv()
        if csv_path is not None:
            return self._load_from_csv(csv_path)
        if settings.data_mode_real:
            return {
                "source": None,
                "classification": "unavailable",
                "path": None,
                "lat": [],
                "lon": [],
                "sea_ice_concentration": [],
                "timestamp_is_utc": True,
                "demo_notice": "Data Unavailable: no real sea-ice CSV found under "
                "Real data/processed/sea_ice/csv/.",
            }
        if PIPELINE_PROCESSED.exists():
            return self._load_from_netcdf(PIPELINE_PROCESSED)
        if self.netcdf_path is not None and self.netcdf_path.exists():
            return self._load_from_netcdf(self.netcdf_path)
        return self._load_synthetic(timestamp)

    def _real_csv(self) -> Path | None:
        """The dedicated latest-snapshot CSV (full grid, newest timestep) if present,
        otherwise the newest time-partitioned file (may be partial for the newest date)."""
        if not REAL_CSV_DIR.is_dir():
            return None
        latest = sorted(REAL_CSV_DIR.glob("sea_ice_latest_part_*.csv"))
        if latest:
            return latest[-1]
        parts = sorted(REAL_CSV_DIR.glob("sea_ice_part_*.csv"))
        return parts[-1] if parts else None

    def _load_from_csv(self, path: Path) -> dict[str, Any]:
        """Pivot the newest timestamp of a processed sea-ice CSV into a grid."""
        try:
            columns = [
                "timestamp", "latitude", "longitude",
                "sea_ice_concentration", "nodata_flag",
            ]
            latest_ts = None
            latest_frames: list[pd.DataFrame] = []
            for chunk in pd.read_csv(path, usecols=columns, chunksize=200_000):
                chunk_ts = chunk["timestamp"].max()
                if latest_ts is None or chunk_ts > latest_ts:
                    latest_ts = chunk_ts
                    latest_frames = [chunk[chunk["timestamp"] == latest_ts]]
                elif chunk_ts == latest_ts:
                    latest_frames.append(chunk[chunk["timestamp"] == latest_ts])
            if not latest_frames:
                raise SeaIceDataError(f"No rows in sea-ice CSV {path}")
            df = pd.concat(latest_frames, ignore_index=True)
        except Exception as exc:
            raise SeaIceDataError(f"Failed to read sea-ice CSV {path}: {exc}") from exc

        if df.empty:
            raise SeaIceDataError(f"No rows in sea-ice CSV {path}")

        # Newest available time step only.
        latest_ts = str(df["timestamp"].iloc[0])

        # Drop nodata/land cells kept for transparency.
        if "nodata_flag" in df.columns:
            df = df.loc[df["nodata_flag"].fillna(0) == 0]

        lat_arr = np.sort(df["latitude"].unique())
        lon_arr = np.sort(df["longitude"].unique())
        if len(lat_arr) == 0 or len(lon_arr) == 0:
            raise SeaIceDataError(f"Sea-ice CSV {path} produced an empty grid")

        grid = np.full((len(lat_arr), len(lon_arr)), np.nan, dtype=float)
        idx = {float(v): i for i, v in enumerate(lat_arr)}
        jdx = {float(v): j for j, v in enumerate(lon_arr)}
        for _, row in df.iterrows():
            i = idx[float(row["latitude"])]
            j = jdx[float(row["longitude"])]
            grid[i, j] = float(row["sea_ice_concentration"])

        return {
            "source": "real_pipeline_csv",
            "classification": "pipeline_data",
            "path": str(path),
            "lat": [float(v) for v in lat_arr],
            "lon": [float(v) for v in lon_arr],
            "sea_ice_concentration": grid.tolist(),
            "timestamp_is_utc": True,
            "timestamp": latest_ts,
        }

    def _load_from_netcdf(self, path: Path) -> dict[str, Any]:
        """Load sea-ice concentration from a NetCDF4 file (pipeline or custom)."""
        try:
            import netCDF4 as nc  # type: ignore

            with nc.Dataset(str(path)) as ds:
                lat_name = "lat" if "lat" in ds.variables else "latitude"
                lon_name = "lon" if "lon" in ds.variables else "longitude"
                lat = np.asarray(ds.variables[lat_name][:], dtype=float)
                lon = np.asarray(ds.variables[lon_name][:], dtype=float)
                sic = np.asarray(ds.variables["sea_ice_concentration"][:], dtype=float)
                if sic.ndim == 3:
                    # Take the newest available time step.
                    sic = sic[-1]

                provenance = {
                    "source": str(ds.source_name if hasattr(ds, "source_name") else "netcdf"),
                    "classification": str(
                        getattr(ds, "classification", "unknown")
                    ),
                }

                return {
                    "source": provenance["source"],
                    "classification": provenance["classification"],
                    "path": str(path),
                    "lat": lat.tolist(),
                    "lon": lon.tolist(),
                    "sea_ice_concentration": sic.tolist(),
                    "timestamp_is_utc": True,
                }
        except ModuleNotFoundError as exc:
            raise SeaIceDataError(
                "netCDF4 package is required to load real NetCDF files. "
                "Install with: pip install netCDF4"
            ) from exc
        except Exception as exc:
            raise SeaIceDataError(f"Failed to read NetCDF file {path}: {exc}") from exc

    def _load_synthetic(self, timestamp: datetime | None = None) -> dict[str, Any]:
        """Generate clearly-labeled synthetic sea-ice concentration data.

        The synthetic field models a smooth ice edge around the Antarctic
        continent for development and testing only.
        """
        ts = timestamp or datetime.now(timezone.utc)

        lat_arr = np.arange(ANTARCTIC_LAT_MIN, ANTARCTIC_LAT_MAX + 0.25, 0.25)
        lon_arr = np.arange(ANTARCTIC_LON_MIN, ANTARCTIC_LON_MAX + 0.25, 0.25)

        seed = int(ts.timestamp()) % 1000
        rng = np.random.RandomState(seed)

        # Ice edge shifts with latitude; nearer the pole => more ice.
        edge = max(-85.0, -62.0 + (seed % 300) / 50.0)

        lats_2d, _ = np.meshgrid(lat_arr, lon_arr, indexing="ij")
        below_pole: np.ndarray = lats_2d < -85.0
        in_ice: np.ndarray = (lats_2d >= -85.0) & (lats_2d < edge)
        in_edge: np.ndarray = (lats_2d >= edge) & (lats_2d < edge + 6.0)

        base = np.zeros_like(lats_2d, dtype=float)
        base[below_pole] = 1.0
        base[in_ice] = 0.95
        base[in_edge] = 0.95 * (1.0 - (lats_2d[in_edge] - edge) / 6.0)

        noise = rng.uniform(-0.05, 0.05, size=lats_2d.shape)
        sic = np.clip(base + noise, 0.0, 1.0)

        return {
            "source": "synthetic_demo",
            "classification": "synthetic_demo",
            "path": None,
            "lat": lat_arr.tolist(),
            "lon": lon_arr.tolist(),
            "sea_ice_concentration": sic.tolist(),
            "timestamp_is_utc": True,
            "demo_notice": "Synthetic sea-ice concentration for development. "
            "Not intended for operational navigation.",
        }

    def save_sample_data(self) -> None:
        """Write the current synthetic field to disk for offline inspection."""
        SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
        data = self._load_synthetic()
        out = SAMPLE_DIR / "sea_ice_concentration_sample.json"
        with open(out, "w") as f:
            json.dump({"meta": data["demo_notice"], "data": data}, f, indent=2)
        return out