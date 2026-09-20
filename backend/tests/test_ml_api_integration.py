"""Integration tests for the ML -> API wiring and navigation hardening.

Covers, all offline and deterministic:

* the read-only model registry (``services.model_registry``),
* the trained-model runtimes for sea-ice / iceberg (``ml.*.runtime``),
  focussing on the honest not-available / error paths,
* the iceberg hard-block (No-Go zone) in the risk engine, and
* land-mask loading (open-water fallback, npy, netCDF).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml.iceberg.runtime import (  # type: ignore[import-not-found]
    _REFERENCE_STEP_HOURS,
    IcebergRuntime,
    get_runtime as get_iceberg_runtime,
)
from ml.sea_ice.runtime import get_runtime as get_sea_ice_runtime  # type: ignore[import-not-found]
from navigation.grid import AntarcticGrid
from navigation.land_mask import load_land_mask
from navigation.risk_engine import RiskConfig, RiskEngine
from services import model_registry


@pytest.fixture(autouse=True)
def _clear_runtime_caches():
    yield
    ml_iceberg_runtime = __import__("ml.iceberg.runtime", fromlist=["_runtimes"])
    ml_sea_ice_runtime = __import__("ml.sea_ice.runtime", fromlist=["_runtimes"])
    ml_iceberg_runtime._runtimes.clear()
    ml_sea_ice_runtime._runtimes.clear()


# ------------------------------------------------------------------ #
# Model registry
# ------------------------------------------------------------------ #

def test_registry_empty_dirs(tmp_path, monkeypatch):
    sea = tmp_path / "sea_ice" / "run"
    ice = tmp_path / "iceberg" / "run"
    sea.mkdir(parents=True)
    ice.mkdir(parents=True)
    monkeypatch.setattr(model_registry, "SEA_ICE_RUN", sea)
    monkeypatch.setattr(model_registry, "ICEBERG_RUN", ice)
    reg = model_registry.model_registry()
    assert reg["models"] == []
    assert reg["count"] == 0
    assert reg["any_real_trained"] is False


def test_registry_reports_demo_artifacts(tmp_path, monkeypatch):
    sea = tmp_path / "sea_ice" / "run"
    ice = tmp_path / "iceberg" / "run"
    sea.mkdir(parents=True)
    ice.mkdir(parents=True)
    (sea / "config.json").write_text(
        json.dumps({"demo": True, "created_utc": "2026-01-01T12:00:00Z"}),
        encoding="utf-8",
    )
    (sea / "model.pt").write_bytes(b"\x00")
    (ice / "run.json").write_text(
        json.dumps({"demo_only": True, "created_utc": "2026-01-02T00:00:00Z"}),
        encoding="utf-8",
    )
    (ice / "rf.joblib").write_bytes(b"\x00")
    monkeypatch.setattr(model_registry, "SEA_ICE_RUN", sea)
    monkeypatch.setattr(model_registry, "ICEBERG_RUN", ice)
    reg = model_registry.model_registry()
    assert reg["count"] == 2
    assert {m["name"] for m in reg["models"]} == {"convlstm", "random_forest"}
    assert all(m["trained_on_demo"] is True for m in reg["models"])
    assert reg["any_real_trained"] is False
    assert "synthetic demo" in (reg["warning"] or "")


def test_registry_flags_real_trained(tmp_path, monkeypatch):
    ice = tmp_path / "iceberg" / "run"
    sea = tmp_path / "sea_ice" / "run"
    ice.mkdir(parents=True)
    sea.mkdir(parents=True)
    (ice / "run.json").write_text(
        json.dumps({"demo_only": False, "created_utc": "2026-01-02T00:00:00Z"}),
        encoding="utf-8",
    )
    (ice / "lstm.pt").write_bytes(b"\x00")
    (sea / "config.json").write_text(json.dumps({"demo": True}), encoding="utf-8")
    (sea / "rf.joblib").write_bytes(b"\x00")
    monkeypatch.setattr(model_registry, "SEA_ICE_RUN", sea)
    monkeypatch.setattr(model_registry, "ICEBERG_RUN", ice)
    reg = model_registry.model_registry()
    assert reg["any_real_trained"] is True
    assert reg["warning"] is None


def test_registry_without_runs_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(model_registry, "SEA_ICE_RUN", tmp_path / "nope")
    monkeypatch.setattr(model_registry, "ICEBERG_RUN", tmp_path / "nope2")
    reg = model_registry.model_registry()
    assert reg["count"] == 0
    assert reg["models"] == []


def test_registry_ignores_corrupt_json(tmp_path, monkeypatch):
    sea = tmp_path / "sea_ice" / "run"
    sea.mkdir(parents=True)
    (sea / "config.json").write_text("{not json", encoding="utf-8")
    (sea / "model.pt").write_bytes(b"\x00")
    monkeypatch.setattr(model_registry, "SEA_ICE_RUN", sea)
    monkeypatch.setattr(model_registry, "ICEBERG_RUN", tmp_path / "empty")
    reg = model_registry.model_registry()
    # corrupt config -> trained_on_demo None but the checkpoint is still listed
    models = [m for m in reg["models"] if m["pipeline"] == "sea_ice"]
    assert models[0]["trained_on_demo"] is None
    assert models[0]["available"] is True


# ------------------------------------------------------------------ #
# Iceberg runtime
# ------------------------------------------------------------------ #

def test_iceberg_runtime_missing_artifacts(tmp_path):
    rt = get_iceberg_runtime("random_forest", out_dir=tmp_path)
    assert rt.available is False
    assert "No trained artifacts" in (rt.error or "")
    assert rt.predict_iceberg("ANY", 24) is None
    assert rt.summary["available"] is False


def test_iceberg_runtime_runjson_missing_with_ckpt(tmp_path):
    (tmp_path / "rf.joblib").write_bytes(b"\x00")
    rt = get_iceberg_runtime("random_forest", out_dir=tmp_path)
    assert rt.available is False
    assert "No trained artifacts" in (rt.error or "")


def test_iceberg_runtime_reports_missing_ckpt(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"demo_only": True}), encoding="utf-8")
    rt = IcebergRuntime(model_kind="lstm", out_dir=tmp_path)
    rt._load()
    assert rt.available is False
    assert "checkpoint" in (rt.error or "").lower()


def test_iceberg_step_hours_nan_guard():
    rt = IcebergRuntime(model_kind="random_forest", out_dir=Path("."))
    assert rt.step_hours_for(None) == _REFERENCE_STEP_HOURS
    assert rt.step_hours_for(_tr(median_dt_hours=float("nan"))) == _REFERENCE_STEP_HOURS
    assert rt.step_hours_for(_tr(median_dt_hours=12.0)) == pytest.approx(12.0)


def _tr(**kwargs):
    class _Fake:
        pass

    t = _Fake()
    for k, v in kwargs.items():
        setattr(t, k, v)
    return t


# ------------------------------------------------------------------ #
# Sea-ice runtime
# ------------------------------------------------------------------ #

def test_sea_ice_runtime_missing_artifacts(tmp_path):
    rt = get_sea_ice_runtime("convlstm", out_dir=tmp_path)
    assert rt.available is False
    assert "No trained artifacts" in (rt.error or "")
    with pytest.raises(RuntimeError, match="unavailable"):
        rt.grid_predict()


def test_sea_ice_runtime_missing_ckpt(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"demo": True, "input_steps": 7, "horizon": 1}),
        encoding="utf-8",
    )
    rt = get_sea_ice_runtime("convlstm", out_dir=tmp_path)
    assert rt.available is False


# ------------------------------------------------------------------ #
# Iceberg hard block (No-Go zone)
# ------------------------------------------------------------------ #

def _grid():
    return AntarcticGrid(-80.0, -60.0, -50.0, 50.0, resolution_deg=2.0)


def test_hard_block_marks_iceberg_zone_non_navigable():
    grid = _grid()
    engine = RiskEngine(grid, RiskConfig(hard_iceberg_block_km=20.0))
    engine.add_icebergs([{"iceberg_id": "IB1", "latitude": -70.0, "longitude": 0.0}])
    engine.apply_iceberg_hard_block()
    assert engine.hard_iceberg_block_applied is True
    assert any("No-Go" in n for n in engine.notes)
    c_hit = grid.cell_for(-70.0, 0.0)
    c_far = grid.cell_for(-70.0, 30.0)
    assert engine.blocked_mask[c_hit.lat_index, c_hit.lon_index]
    assert not engine.blocked_mask[c_far.lat_index, c_far.lon_index]
    assert engine.obstacle_mask()[c_hit.lat_index, c_hit.lon_index]


def test_hard_block_sets_total_risk_1_inside_zone():
    grid = _grid()
    engine = RiskEngine(grid, RiskConfig(hard_iceberg_block_km=20.0))
    engine.add_icebergs([{"iceberg_id": "IB1", "latitude": -70.0, "longitude": 0.0}])
    engine.apply_iceberg_hard_block()
    engine.compute_total()
    assert engine.risk_at(-70.0, 0.0) == pytest.approx(1.0)


def test_hard_block_mutually_exclusive_with_soft_use():
    """With hard block off, iceberg is only a weighted risk, not blocked."""
    grid = _grid()
    engine = RiskEngine(grid, RiskConfig(hard_iceberg_block_km=None))
    engine.add_icebergs([{"iceberg_id": "IB1", "latitude": -70.0, "longitude": 0.0}])
    engine.apply_iceberg_hard_block()
    assert engine.hard_iceberg_block_applied is False
    c = grid.cell_for(-70.0, 0.0)
    assert not engine.blocked_mask[c.lat_index, c.lon_index]


# ------------------------------------------------------------------ #
# Land-mask loading
# ------------------------------------------------------------------ #

def test_land_mask_no_config_is_open_water():
    grid = _grid()
    mask, info = load_land_mask(grid, None)
    assert mask is None
    assert info["loaded"] is False
    assert "open water" in info["reason"] or "LAND_MASK_FILE" in info["reason"]


def test_land_mask_missing_file():
    grid = _grid()
    mask, info = load_land_mask(grid, str(Path("C:/definitely/not/here.nc")))
    assert mask is None
    assert info["loaded"] is False


def test_land_mask_npy_aligned(tmp_path):
    grid = _grid()
    mask = np.zeros((grid.nlat, grid.nlon), dtype=bool)
    mask[5, 7] = True
    path = tmp_path / "land.npy"
    np.save(path, mask)
    loaded, info = load_land_mask(grid, str(path))
    assert loaded is not None
    assert info["loaded"] is True
    assert info["format"] == "npy"
    assert info["land_cells"] == 1
    assert loaded[5, 7]
    assert not loaded[0, 0]


def test_land_mask_npy_misaligned(tmp_path):
    grid = _grid()
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros((5, 5), dtype=bool))
    mask, info = load_land_mask(grid, str(path))
    assert mask is None
    assert info["loaded"] is False
    assert "shape" in info["reason"]


def test_land_mask_netcdf_elevation(tmp_path):
    xr = pytest.importorskip("xarray")
    grid = _grid()
    lat = grid.lats
    lon = grid.lons
    elev = np.full((len(lat), len(lon)), -500.0)
    elev[0, 0] = 800.0
    elev[grid.nlat // 2, grid.nlon // 2] = 800.0
    ds = xr.Dataset({"elevation": (("lat", "lon"), elev)}, coords={"lat": lat, "lon": lon})
    path = tmp_path / "land.nc"
    ds.to_netcdf(path)
    loaded, info = load_land_mask(grid, str(path))
    assert loaded is not None
    assert info["loaded"] is True
    assert info["format"] == "netcdf"
    assert info["land_cells"] == 2  # exactly the two +800 cells survive regridding
    c0 = grid.cell_for(lat[0], lon[0])
    c1 = grid.cell_for(lat[grid.nlat // 2], lon[grid.nlon // 2])
    assert loaded[c0.lat_index, c0.lon_index]
    assert loaded[c1.lat_index, c1.lon_index]


def test_land_mask_integration_into_risk_engine(tmp_path):
    grid = _grid()
    np.save(tmp_path / "land.npy", np.eye(grid.nlat, grid.nlon, dtype=bool))
    loaded, info = load_land_mask(grid, str(tmp_path / "land.npy"))
    assert info["loaded"] is True
    grid.set_land_mask(loaded)
    engine = RiskEngine(grid, RiskConfig())
    engine.compute_total()
    c_diag = grid.cell_for(-80.0, -50.0)  # first grid cell (northwest on the diagonal)
    assert not grid.is_navigable(c_diag)
    assert engine.total_risk[c_diag.lat_index, c_diag.lon_index] == pytest.approx(1.0)