"""Tests for the data pipeline: demo generation, imports, validation, accessor.

All tests run offline. They exercise synthetic demo data only and never
touch real weather/sea-ice providers.
"""
from __future__ import annotations

import numpy as np
import pytest

from data_pipeline.demo.create import create_all_demo_data
from data_pipeline.icebergs.import_data import import_file, normalize_to_pandas
from data_pipeline.icebergs.import_data import validate_processed as validate_icebergs
from data_pipeline.ocean.acquire import validate as validate_ocean
from data_pipeline.processor.accessor import EnvironmentalAccessor
from data_pipeline.sea_ice.acquire import validate as validate_sea_ice
from data_pipeline.weather.acquire import validate as validate_weather


@pytest.fixture(scope="module")
def pipeline_root(tmp_path_factory):
    return tmp_path_factory.mktemp("pipeline")


@pytest.fixture(scope="module")
def demo_outputs(pipeline_root):
    return create_all_demo_data(pipeline_root, days=7)


def test_demo_creates_all_datasets(demo_outputs):
    for name, paths in demo_outputs.items():
        for p in paths:
            assert p.exists(), f"{name} missing: {p}"


@pytest.mark.parametrize("name", ["sea_ice", "ocean", "weather"])
def test_demo_netcdf_attributes_are_demo(demo_outputs, name):
    import xarray as xr

    path = demo_outputs[name][0]
    ds = xr.open_dataset(path)
    assert ds.attrs.get("classification") == "synthetic_demo"
    assert "time" in ds.coords


def test_validators_pass_on_demo_data(demo_outputs, pipeline_root):
    sea_ice_result = validate_sea_ice(pipeline_root)
    assert sea_ice_result.status == "valid"
    assert any(c.severity == "warning" for c in sea_ice_result.checks)  # demo warning

    assert validate_ocean(pipeline_root).status == "valid"
    assert validate_weather(pipeline_root).status == "valid"
    assert validate_icebergs(pipeline_root).status == "valid"


def test_iceberg_import_from_csv(demo_outputs, pipeline_root):
    csv = pipeline_root / "icebergs_demo.csv"
    result = import_file(csv, pipeline_root, tag="icebergs_demo")
    assert result.status == "valid"
    assert (pipeline_root / "icebergs_demo.csv").exists()

    nc = pipeline_root / "icebergs_demo.nc"
    assert nc.exists(), "NetCDF output missing"


def test_iceberg_normalize_km_conversion(pipeline_root):
    import pandas as pd

    df = pd.DataFrame(
        {
            "id": ["X1", "X1"],
            "time": ["2026-01-01", "2026-01-02"],
            "lat": [-65.0, -64.9],
            "lon": [-100.0, -99.9],
            "length_km": [10.0, 10.5],
            "width_km": [5.0, 5.2],
        }
    )
    norm = normalize_to_pandas(df)
    assert norm.iloc[0]["length_nm"] == pytest.approx(10.0 * 0.539957)
    assert norm.iloc[0]["width_nm"] == pytest.approx(5.0 * 0.539957)


def test_accessor_freshness_and_interpolation(demo_outputs, pipeline_root):
    accessor = EnvironmentalAccessor(pipeline_root)
    report = accessor.freshness_report()
    by_name = {e["dataset"]: e for e in report}
    assert by_name["sea_ice"]["cadence_hours"] == 24.0
    assert by_name["sea_ice"]["classification"] == "synthetic_demo"

    import xarray as xr

    times = xr.open_dataset(pipeline_root / "sea_ice.nc").time.values
    query_time = times[len(times) // 2]
    res = accessor.query(
        "sea_ice", "sea_ice_concentration", query_time, method="nearest"
    )
    assert res["metadata"]["method"] == "exact_match"
    assert res["values"].ndim == 2
    assert bool(res["metadata"]["is_demo"])


def test_accessor_linear_interpolation(demo_outputs, pipeline_root):
    accessor = EnvironmentalAccessor(pipeline_root)
    import xarray as xr

    ds = xr.open_dataset(pipeline_root / "sea_ice.nc")
    if len(ds.time) < 2:
        pytest.skip("needs >=2 timestamps for interpolation")
    t0 = ds.time.values[0]
    t1 = ds.time.values[1]
    mid = t0 + np.timedelta64(6, "h")
    res = accessor.query(
        "sea_ice", "sea_ice_concentration", mid, method="linear"
    )
    assert res["metadata"]["method"] == "linear_in_time"
    assert res["metadata"]["bracketing_times"][0] != res["metadata"]["bracketing_times"][1]
    assert 0.0 < res["metadata"]["interpolation_weight"] < 1.0