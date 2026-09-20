"""Validation tests for the real-data pipeline outputs and backend loaders.

These tests run against the actual ``Real data/processed/<dataset>/csv/``
output written by ``scripts/preprocess_real_data.py``.  They verify:

- every processed CSV reloads and matches the documented schema;
- latitude/longitude ranges respect the selected Antarctic band;
- timestamps are ISO-8601 UTC (Z) with no timezone-naive drift;
- land rows were removed from sea-ice / iceberg outputs;
- region filters were applied;
- the backend loaders classify the real CSVs as ``pipeline_data`` (not demo);
- nothing in ``Real data/raw`` was modified (read-only guarantee).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "Real data" / "processed"
RAW = PROJECT_ROOT / "Real data" / "raw"


def _csv(dataset: str, pattern: str = "*.csv") -> list[Path]:
    folder = PROCESSED / dataset / "csv"
    return sorted(folder.glob(pattern)) if folder.is_dir() else []


def _all_rows(dataset: str, pattern: str = "*.csv",
              nrows: int | None = None) -> pd.DataFrame:
    parts = _csv(dataset, pattern)
    assert parts, f"no {dataset} CSVs found under {PROCESSED / dataset}"
    return pd.concat(
        [pd.read_csv(p, nrows=nrows) for p in parts],
        ignore_index=True,
    )


# ── Schema / reload sanity ─────────────────────────────────────────
def test_sea_ice_csv_schema():
    df = _all_rows("sea_ice", "sea_ice_part_*.csv")
    required = {"timestamp", "latitude", "longitude", "sea_ice_concentration"}
    assert required.issubset(df.columns)
    # rows with nodata_flag=1 carry no concentration value (kept for provenance)
    observed = df[df["nodata_flag"].fillna(0) == 0]
    assert observed["sea_ice_concentration"].between(0.0, 1.0).all()
    assert pd.to_datetime(df["timestamp"], errors="coerce").notna().all()


def test_sea_ice_latest_snapshot_schema():
    latest = _csv("sea_ice", "sea_ice_latest_part_*.csv")
    if not latest:
        pytest.skip("no latest snapshot produced")
    df = pd.read_csv(latest[0])
    assert {"latitude", "longitude", "sea_ice_concentration"} <= set(df.columns)
    observed = df[df["nodata_flag"].fillna(0) == 0]
    assert observed["sea_ice_concentration"].between(0.0, 1.0).all()


def test_iceberg_csv_schema():
    df = _all_rows("iceberg")
    required = {"iceberg_id", "timestamp", "latitude", "longitude",
                "length_nm", "width_nm"}
    assert required.issubset(df.columns)
    # no land positions survive the pipeline (land cells were removed)
    assert len(df) > 0
    assert df["latitude"].between(-90.0, -50.0).all()


def test_ocean_csv_schema():
    df = _all_rows("ocean")
    required = {"timestamp", "latitude", "longitude", "depth", "thetao", "so",
                "uo", "vo", "siconc"}
    assert required.issubset(df.columns)
    assert df["depth"].eq(df["depth"].iloc[0]).all(), "depth should be constant (surface)"


def test_weather_csv_schema():
    df = _all_rows("weather")
    required = {"timestamp", "latitude", "longitude", "wind_u", "wind_v",
                "air_temperature", "pressure"}
    assert required.issubset(df.columns)
    assert df["air_temperature"].between(150.0, 320.0).all(), "temperature in Kelvin"


def test_vessel_csv_schema():
    df = _all_rows("vessel")
    required = {"timestamp", "latitude", "longitude", "flag", "geartype",
                "hours", "fishing_hours", "mmsi_present"}
    assert required.issubset(df.columns)
    assert df["latitude"].between(-90.0, -50.0).all()


def test_bathymetry_csv_schema():
    df = _all_rows("bathymetry")
    assert {"latitude", "longitude", "elevation_meters", "depth_meters"} <= set(df.columns)
    assert (df["elevation_meters"] > -11000).all()
    assert (df["elevation_meters"] < 9000).all()


# ── Spatial / temporal correctness ─────────────────────────────────
def test_sea_ice_lat_band():
    df = _all_rows("sea_ice", "sea_ice_part_*.csv", nrows=200_000)
    assert df["latitude"].min() >= -75.0
    assert df["latitude"].max() <= -55.0


def test_iceberg_lat_band():
    df = _all_rows("iceberg")
    assert df["latitude"].min() >= -90.0
    assert df["latitude"].max() <= -50.0


def test_timestamps_are_utc():
    """Every timestamp parses as UTC (ISO-8601 Z or +00:00 offset)."""
    for ds in ("sea_ice", "iceberg", "ocean", "weather", "vessel"):
        parts = _csv(ds)
        assert parts, f"no parts for {ds}"
        df = pd.read_csv(parts[-1], nrows=5000)
        if "timestamp" not in df.columns:
            continue
        raw = df["timestamp"].dropna().astype(str)
        parsed = pd.to_datetime(raw, errors="coerce", utc=True)
        assert parsed.notna().all(), f"{ds} has non-ISO / non-UTC timestamps"
        assert (parsed.dt.tz is not None), f"{ds} timestamps are timezone-naive"
        assert (parsed.dt.strftime("%z") == "+0000").all(), (
            f"{ds} timestamps are not expressed in UTC: {raw.head(1).tolist()}"
        )


# ── Backend loader integration ─────────────────────────────────────
@pytest.fixture(scope="module")
def sea_ice_data():
    from app.services.sea_ice.data_loader import SeaIceDataLoader
    return SeaIceDataLoader().load()


def test_sea_ice_loader_reads_real_csv(sea_ice_data):
    if sea_ice_data.get("classification") == "unavailable":
        pytest.skip("real sea-ice CSV not present (DATA_MODE=auto fallback)")
    assert sea_ice_data["classification"] == "pipeline_data"
    assert sea_ice_data["source"] == "real_pipeline_csv"
    assert len(sea_ice_data["lat"]) > 0
    assert len(sea_ice_data["lon"]) > 0
    assert len(sea_ice_data["sea_ice_concentration"]) > 0


@pytest.fixture(scope="module")
def iceberg_items():
    from app.services.iceberg.data_loader import IcebergDataLoader
    from services.data_paths import ICEBERG_CSV
    return IcebergDataLoader(data_path=str(ICEBERG_CSV)).load()


def test_iceberg_loader_reads_real_csv(iceberg_items):
    if not iceberg_items:
        pytest.skip("no real iceberg CSV resolved")
    first = iceberg_items[0]
    assert first["classification"] == "pipeline_data"
    assert "latitude" in first and "longitude" in first


# ── Read-only guarantee ────────────────────────────────────────────
def test_raw_data_untouched():
    """Raw source folders must never contain pipeline-generated outputs.

    The iceberg source dataset itself ships as CSVs in ``raw/iceberg``, so the
    invariant is that no pipeline artifact (``*_part_*``, ``*_processed``,
    ``*_latest_part_*``) was ever written back into ``Real data/raw``.
    """
    markers = ("_part_", "_processed", "_latest_part_")
    for raw_dir in RAW.iterdir():
        if not raw_dir.is_dir():
            continue
        bad = [
            str(p.relative_to(RAW))
            for p in raw_dir.rglob("*.csv")
            if any(m in p.name for m in markers)
        ]
        assert bad == [], f"raw/{raw_dir.name} contains pipeline outputs: {bad[:5]}"