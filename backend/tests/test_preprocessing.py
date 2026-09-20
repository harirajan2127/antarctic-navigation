"""Tests for the data pre-processing pipeline and feature engineering.

All runs offline using synthetic demo data.  Never touches real providers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.features.feature_engineering import FeatureEngineering
from data.processing._common import EnvLookup, forward_bearing, haversine_nm
from data.processing.iceberg_processor import IcebergProcessor
from data.processing.ocean_processor import OceanProcessor
from data.processing.sea_ice_processor import SeaIceProcessor
from data.processing.weather_processor import WeatherProcessor


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("preproc")
    processed = root / "processed"
    reports = root / "reports"
    processed.mkdir()
    reports.mkdir()
    return root, processed, reports


@pytest.fixture(scope="module")
def pipeline_demo(workspace):
    """Create demo inputs once, at module scope, mirroring run_preprocessing."""
    root, processed, reports = workspace
    demo = Path("datasets/processed")

    sea_ice_out = processed / "sea_ice" / "sea_ice_clean.nc"
    ocean_out = processed / "ocean" / "ocean_clean.nc"
    weather_out = processed / "weather" / "weather_clean.nc"
    iceberg_csv = processed / "icebergs" / "icebergs_clean.csv"
    iceberg_nc = processed / "icebergs" / "icebergs_clean.nc"

    SeaIceProcessor(
        demo / "sea_ice.nc", sea_ice_out, reports,
    ).process()
    OceanProcessor(demo / "ocean_surface.nc", ocean_out, reports).process()
    WeatherProcessor(demo / "weather_surface.nc", weather_out, reports).process()
    IcebergProcessor(
        demo / "icebergs_demo.csv", iceberg_csv, iceberg_nc, reports
    ).process()

    return root, processed, reports, sea_ice_out, ocean_out, weather_out, iceberg_csv


def test_sea_ice_processor_output(workspace, pipeline_demo):
    root, processed, reports, sea_ice_out, *_ = pipeline_demo
    import xarray as xr

    ds = xr.open_dataset(sea_ice_out)
    assert "sea_ice_concentration" in ds
    assert ds["latitude"].min() >= -90 and ds["latitude"].max() <= -50
    svals = ds["sea_ice_concentration"].values
    assert np.nanmin(svals) >= 0.0 and np.nanmax(svals) <= 1.0
    assert (reports / "sea_ice_processing.json").exists()
    assert (reports / "sea_ice_processing.md").exists()
    ds.close()


def test_ocean_processor_variables(pipeline_demo):
    import xarray as xr

    _, _, _, _, ocean_out, *_ = pipeline_demo
    ds = xr.open_dataset(ocean_out)
    for var in ("uo", "vo", "thetao", "so"):
        assert var in ds, f"missing {var}"
        assert ds[var].attrs.get("units") in ("m s-1", "K", "psu")
    ds.close()


def test_weather_processor_units(pipeline_demo):
    import xarray as xr

    _, _, _, _, _, weather_out, _ = pipeline_demo
    ds = xr.open_dataset(weather_out)
    units = {v: ds[v].attrs.get("units") for v in ("u10", "v10", "t2m", "msl")}
    assert units["u10"] == "m s-1"
    assert units["v10"] == "m s-1"
    assert units["t2m"] == "K"
    assert units["msl"] == "Pa"
    ds.close()


def test_iceberg_kinematics(pipeline_demo):
    _, _, _, _, _, _, iceberg_csv = pipeline_demo
    df = pd.read_csv(iceberg_csv, parse_dates=["timestamp"])
    assert df["iceberg_id"].nunique() > 0
    # history preserved: each iceberg's records sorted in time
    per_iceberg_sorted = df.groupby("iceberg_id")["timestamp"].apply(
        lambda ts: ts.is_monotonic_increasing
    )
    assert per_iceberg_sorted.all()
    # history preserved: same iceberg sorted in time
    one = df[df["iceberg_id"] == df["iceberg_id"].iloc[0]].sort_values("timestamp")
    assert len(one) >= 2
    # displacement consistent with haversine
    d = haversine_nm(
        one["latitude"].iloc[:-1].to_numpy(),
        one["longitude"].iloc[:-1].to_numpy(),
        one["latitude"].iloc[1:].to_numpy(),
        one["longitude"].iloc[1:].to_numpy(),
    )
    assert np.allclose(d, one["displacement_nm"].iloc[1:].to_numpy(), atol=0.01)


def test_forward_bearing_known():
    import numpy as np

    # due east from origin
    assert forward_bearing(np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(10.0)) == pytest.approx(90.0, abs=0.1)
    # due north
    assert forward_bearing(np.float64(0.0), np.float64(0.0), np.float64(10.0), np.float64(0.0)) == pytest.approx(0.0, abs=0.1)


def test_feature_engineering_columns_and_lag(pipeline_demo):
    _, processed, reports, sea_ice, ocean, weather, iceberg_csv = pipeline_demo
    fe = FeatureEngineering(
        iceberg_csv, sea_ice, ocean, weather,
        processed / "features", reports, tolerance_hours=24.0,
    )
    df = fe.build()
    expected = {
        "sic_lag", "sic_change", "current_magnitude", "current_direction",
        "wind_magnitude", "wind_direction", "prev_latitude", "prev_longitude",
        "iceberg_velocity_knots", "hour_of_day", "day_of_year", "month",
        "sea_ice_gap_hours", "ocean_gap_hours", "weather_gap_hours",
        "sea_ice_stale", "ocean_stale", "weather_stale",
    }
    assert expected.issubset(set(df.columns))
    # lag: first row of each iceberg has NaN sic_lag/prev pos
    first = df.groupby("iceberg_id").head(1)
    assert first["sic_lag"].isna().all()
    # stale guard satisfied within tolerance
    assert df[["sea_ice_stale", "ocean_stale", "weather_stale"]].astype(bool).any(axis=1).sum() == 0
    # splits are strictly chronological
    splits = fe.split_chronological(df)
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == len(df)
    assert splits["train"]["timestamp"].max() <= splits["val"]["timestamp"].min()
    assert splits["val"]["timestamp"].max() <= splits["test"]["timestamp"].min()


def test_no_silent_join_with_tight_tolerance(pipeline_demo):
    """At a 1-hour tolerance all env rows must be stale (grids are >=6h)."""
    _, processed, reports, sea_ice, ocean, weather, iceberg_csv = pipeline_demo
    fe = FeatureEngineering(
        iceberg_csv, sea_ice, ocean, weather,
        processed / "features_strict", reports, tolerance_hours=1.0,
    )
    df = fe.build()
    stale = df[["sea_ice_stale", "ocean_stale", "weather_stale"]].astype(bool).any(axis=1)
    assert stale.sum() == len(df)
    # stale rows carry NaN env fields
    assert df.loc[stale, "sea_ice_concentration"].isna().all()


def test_env_lookup_metadata():
    lookup = EnvLookup(Path("datasets/processed/sea_ice.nc"), ("sea_ice_concentration",))
    assert set(lookup.data) == {"sea_ice_concentration"}
    res = lookup.lookup(-65.0, 0.0, lookup.times[len(lookup.times) // 2], tolerance_hours=1)
    assert "gap_hours" in res and "stale" in res
    assert res["stale"] is False or res["stale"] is True


def test_reports_written(pipeline_demo):
    reports = pipeline_demo[2]
    expected = {
        "sea_ice_processing.json", "ocean_processing.json", "weather_processing.json",
        "iceberg_processing.json", "feature_summary.json", "dataset_shape.json",
        "unit_report.json",
    }
    present = {p.name for p in reports.iterdir()}
    assert expected.issubset(present)