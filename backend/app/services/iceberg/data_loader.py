"""Iceberg data loading from pipeline output, files, or labeled synthetic data.

Priority:
1. Pipeline-processed CSV (``datasets/processed/icebergs.csv`` when present).
2. Configured JSON/GeoJSON path.
3. Clearly-labeled synthetic demo icebergs.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from config import settings

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "data" / "sample"
PIPELINE_PROCESSED = Path(__file__).resolve().parents[2] / "datasets" / "processed" / "icebergs.csv"
PIPELINE_DEMO = Path(__file__).resolve().parents[2] / "datasets" / "processed" / "icebergs_demo.csv"

NM_TO_KM = 1.852


class IcebergDataError(Exception):
    """Raised when iceberg data cannot be loaded."""


# ---------------------------------------------------------------------------
# Coastal band annotation (dashboard display only)
# ---------------------------------------------------------------------------
# The dashboard map should only draw icebergs that sit in the sea and nearest
# to the Antarctic coast.  To decide where the coast is we use the *same* land
# mask the rest of the pipeline uses: the processed ocean grid, where ``thetao``
# is NaN on land cells (and finite on sea cells).  We attach two additive
# fields to every loaded iceberg:
#
#   ``on_land``             -> True when the iceberg sits on a land cell
#   ``distance_to_coast_km``-> great-circle distance to the nearest land cell
#
# Nothing else changes: positions, counts, predictions and analytics consume
# the identical list they did before — only the dashboard's map markers use
# these two new fields to keep at-sea / nearest-to-coast icebergs visible and
# hide the ones currently drawn on the continent.
_OCEAN_GRID = Path(__file__).resolve().parents[3] / "data" / "processed" / "ocean" / "ocean_clean.nc"


class _LandMask:
    """Lazily-loaded land mask + nearest-land distance from the ocean grid.

    Thread-safe enough for dev/demo: the grid is read once and cached module
    wide; repeated lookups are vectorised over the masked land cells.
    """

    _loaded = False
    _land_lat: np.ndarray | None = None
    _land_lon: np.ndarray | None = None
    _cell_km: float = 0.0

    @classmethod
    def _ensure(cls) -> None:
        if cls._loaded:
            return
        cls._loaded = True
        if not _OCEAN_GRID.exists():
            return
        try:
            import xarray as xr
        except ImportError:
            return
        try:
            with xr.open_dataset(_OCEAN_GRID) as ds:
                lat = np.asarray(ds.latitude.values, dtype=float)
                lon = np.asarray(ds.longitude.values, dtype=float)
                th = ds.thetao.values
        except OSError:
            return
        if th.ndim == 3:
            th = th[-1]
        land = np.isnan(th)
        if not land.any():
            return
        lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
        cls._land_lat = lat2[land].ravel()
        cls._land_lon = lon2[land].ravel()
        cls._cell_km = float(2.0 * 6371.0 * np.pi * (np.abs(lat[1] - lat[0])) / 360.0)

    @classmethod
    def _distance_km(cls, lat: float, lon: float) -> float | None:
        cls._ensure()
        if cls._land_lat is None or cls._land_lat.size == 0:
            return None
        dlat = np.radians(cls._land_lat - lat)
        dlon = np.radians(cls._land_lon - lon)
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(np.radians(lat))
            * np.cos(np.radians(cls._land_lat))
            * np.sin(dlon / 2.0) ** 2
        )
        d = 6371.0 * 2.0 * np.arcsin(np.sqrt(np.minimum(a, 1.0)))
        return float(d.min())

    @classmethod
    def annotate(cls, item: dict[str, Any]) -> dict[str, Any]:
        """Add ``on_land`` and ``distance_to_coast_km`` (additive, best effort)."""
        d = cls._distance_km(float(item["latitude"]), float(item["longitude"]))
        item["on_land"] = d is not None and d <= cls._cell_km
        item["distance_to_coast_km"] = None if d is None else round(d, 1)
        return item


class IcebergDataLoader:
    """Loads iceberg track data for trajectory prediction."""

    def __init__(self, data_path: str | None = None) -> None:
        self.data_path = Path(data_path) if data_path else None

    def load(self, timestamp: datetime | None = None) -> list[dict[str, Any]]:
        """Load iceberg positions and histories.

        Prefers real processed CSVs (``Real data/processed/iceberg/csv/``) when
        present, then the pipeline-processed dataset, then the configured file
        path, then labeled synthetic demo data. When ``DATA_MODE=real`` and no
        real data exists, returns an empty list (the API reports
        *Data Unavailable*).
        """
        candidates = [
            p
            for p in (
                self._real_csv() or self._real_default_csv(),
                PIPELINE_PROCESSED,
                PIPELINE_DEMO,
                self.data_path,
            )
            if p is not None and p.exists()
        ]
        if candidates:
            items = self._load_from_pipeline(candidates[0])
            return [_LandMask.annotate(item) for item in items]
        if settings.data_mode_real:
            return []
        items = self._load_synthetic(timestamp)
        return [_LandMask.annotate(item) for item in items]

    def _real_csv(self) -> Path | None:
        """Confirm a real, non-demo CSV was explicitly configured."""
        if self.data_path is None:
            return None
        if "demo" in Path(self.data_path).stem:
            return None
        return self.data_path if Path(self.data_path).is_file() else None

    def _real_default_csv(self) -> Path | None:
        """Real iceberg CSV from ``Real data/processed/iceberg/csv`` in real mode."""
        if not settings.data_mode_real:
            return None
        folder = Path(__file__).resolve().parents[4] / "Real data" / "processed" / "iceberg" / "csv"
        if not folder.is_dir():
            return None
        files = sorted(p for p in folder.glob("*.csv") if p.is_file())
        return files[0] if files else None

    def _load_from_pipeline(self, path: Path) -> list[dict[str, Any]]:
        """Load iceberg records from a pipeline-processed CSV/NetCDF file.

        Returns only the newest observation per iceberg, expressed in the
        app's km-based schema, tagged with provenance.
        """
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            raise IcebergDataError(
                "pandas is required to load pipeline-processed iceberg data."
            ) from exc

        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        latest = df.groupby("iceberg_id").last().reset_index()

        is_demo = "demo" in Path(path).stem
        items = []
        for _, row in latest.iterrows():
            items.append(
                {
                    "iceberg_id": str(row["iceberg_id"]),
                    "latitude": round(float(row["latitude"]), 4),
                    "longitude": round(float(row["longitude"]), 4),
                    "length_km": round(float(row["length_nm"]) * NM_TO_KM, 2),
                    "width_km": round(float(row["width_nm"]) * NM_TO_KM, 2),
                    "last_observed": str(row["timestamp"]),
                    "source": Path(path).name,
                    "classification": "synthetic_demo" if is_demo else "pipeline_data",
                    **(
                        {"demo_notice": "Synthetic iceberg data for development only."}
                        if is_demo
                        else {}
                    ),
                }
            )
        return items

    def _load_from_file(self, path: Path) -> list[dict[str, Any]]:
        """Load iceberg data from a JSON or GeoJSON file."""
        with open(path, "r") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            return raw
        features = raw.get("features", [])
        result = []
        for feature in features:
            geom = feature.get("geometry", {})
            props = feature.get("properties", {})
            coords = geom.get("coordinates", [None, None])
            if coords and coords[0] is not None:
                result.append(
                    {
                        "iceberg_id": props.get("id", "unknown"),
                        "latitude": coords[1],
                        "longitude": coords[0],
                        "length_km": props.get("length_km", 0),
                        "width_km": props.get("width_km", 0),
                    }
                )
        return result

    def _load_synthetic(self, timestamp: datetime | None = None) -> list[dict[str, Any]]:
        """Generate synthetic icebergs around the Antarctic margin for development."""
        ts = timestamp or datetime.now(timezone.utc)
        seed = int(ts.timestamp()) % 10000
        rng = np.random.RandomState(seed)

        icebergs = []
        for i in range(40):
            lat = rng.uniform(-78.0, -58.0)
            lon = rng.uniform(-180.0, 180.0)
            drift = rng.normal(1.0, 0.5)  # km/hr drift for demo
            icebergs.append(
                {
                    "iceberg_id": f"DEMO-A{i:03d}",
                    "latitude": round(float(lat), 4),
                    "longitude": round(float(lon), 4),
                    "length_km": round(float(rng.uniform(0.5, 30.0)), 2),
                    "width_km": round(float(rng.uniform(0.3, 25.0)), 2),
                    "drift_speed_km_per_hour": round(float(drift), 3),
                    "drift_direction_deg": round(float(rng.uniform(0, 360)), 1),
                    "demo_notice": "Synthetic iceberg data for development only.",
                }
            )
        return icebergs

    def save_sample_data(self) -> None:
        """Write synthetic iceberg data to disk for offline inspection."""
        SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
        data = self._load_synthetic()
        out = SAMPLE_DIR / "icebergs_sample.json"
        with open(out, "w") as f:
            json.dump(
                {"meta": "Synthetic iceberg data for development only.", "data": data},
                f,
                indent=2,
            )
        return out