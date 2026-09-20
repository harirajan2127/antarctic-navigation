"""Shared navigation engine assembly for the route API and journey simulator.

Both the modern ``POST /api/v1/routes/optimize`` endpoint, the persistence
service and the ``/api/v1/navigation`` journey simulator must produce the
*humanly same* risk grid from the *same* configured data sources, vessel
constraints and hazard rules. This module owns that assembly so the two code
paths can never drift apart:

    sea-ice (real CSV -> persistence forecast)
  + icebergs (real observations + predictions, ``ICEBERG_MODEL``)
  + weather severity (newest real wind field, or an honest neutral baseline)
  + vessel constraints (ice class, clearance)
  + hard no-go zones: iceberg block radius, sea-ice block concentration
  + land mask (kept fully active; points on land resolve to an ocean approach)

Each ``build_route_engine`` call rebuilds the grid from the latest data
snapshot, exactly like the pre-existing per-call engine builds.
"""
from __future__ import annotations

from datetime import datetime, timezone
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.services.sea_ice.data_loader import SeaIceDataLoader
from app.services.sea_ice.forecaster import SeaIceForecastService
from config import DEFAULT_LAND_MASK_FILE, settings
from navigation.grid import AntarcticGrid
from navigation.land_mask import load_land_mask
from navigation.risk_engine import RiskConfig, RiskEngine
from services.iceberg_service import iceberg_service

logger = __import__("logging").getLogger("dss.services.route_engine")

_WEATHER_SEVERITY_CACHE: dict[str, tuple[np.ndarray, bool]] = {}

# Short-TTL cache of the *raw data sources* used to assemble a risk engine.
# The fingerprint is derived from the actual data files on disk (path + mtime +
# size), so new data invalidates the cache naturally. The assembled engine is
# always re-derived per call: engine state is per-request and never shared.
_SOURCE_CACHE_TTL_SECONDS = 120.0
_SOURCE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_LAND_MASK_CACHE: dict[str, tuple[np.ndarray | None, dict[str, Any]]] = {}


def _source_fingerprint() -> str:
    """Fingerprint the currently-configured data-source files on disk."""
    parts: list[str] = []

    def _add(path: Path) -> None:
        try:
            if path.is_file():
                st = path.stat()
                parts.append(f"{path}:{st.st_mtime_ns}:{st.st_size}")
            elif path.is_dir():
                files = sorted(
                    (p for p in path.glob("**/*") if p.is_file()), key=lambda p: str(p)
                )
                for p in files[-3:]:
                    st = p.stat()
                    parts.append(f"{p}:{st.st_mtime_ns}:{st.st_size}")
        except OSError:
            parts.append(str(path))

    for each in (
        Path(__file__).resolve().parents[2] / "Real data" / "processed" / "sea_ice" / "csv",
        Path(__file__).resolve().parents[2] / "Real data" / "processed" / "iceberg" / "csv",
        Path(__file__).resolve().parents[1] / "data" / "processed" / "features" / "feature_table.csv",
        Path(__file__).resolve().parents[2] / "Real data" / "processed" / "weather" / "csv",
    ):
        _add(each)
    return "|".join(parts)


def _load_source_snapshot(ts: datetime, config: dict[str, Any]) -> dict[str, Any]:
    """Load and lightly cache the raw data sources for engine assembly."""
    started = time.perf_counter()
    horizon = int(config.get("forecast_horizon_hours", 24))
    fingerprint = _source_fingerprint()

    cached = _SOURCE_CACHE.get(fingerprint)
    now = __import__("time").time()
    if cached is not None and (now - cached[0]) < _SOURCE_CACHE_TTL_SECONDS:
        logger.info("ROUTE_DATA_LOADED cache_hit=true elapsed_ms=%.1f", (time.perf_counter() - started) * 1000)
        return dict(cached[1], _horizon=horizon)

    logger.info("ROUTE_DATA_LOADED cache_hit=false")
    sea_ice_loader = SeaIceDataLoader(netcdf_path=None)
    sea_ice = sea_ice_loader.load(ts)
    logger.info(
        "SEA_ICE_DATA_LOADED classification=%s lat=%d lon=%d",
        sea_ice.get("classification", "unknown"),
        len(sea_ice.get("lat", [])),
        len(sea_ice.get("lon", [])),
    )
    classification = str(sea_ice.get("classification", "unknown"))
    forecast = SeaIceForecastService(settings.SEA_ICE_MODEL or "persistence").forecast(
        sea_ice["sea_ice_concentration"],
        sea_ice["lat"],
        sea_ice["lon"],
        timestamp=ts,
        horizon_hours=horizon,
    )

    iceberg_data = iceberg_service.predict(
        None, horizon_hours=horizon, model=settings.ICEBERG_MODEL or "persistence"
    )
    logger.info(
        "ICEBERG_DATA_LOADED classification=%s count=%d model=%s",
        iceberg_data.get("classification", "unknown"),
        len(iceberg_data.get("icebergs", [])),
        iceberg_data.get("model", "unknown"),
    )

    weather_severity, weather_ok = _real_weather_severity(
        AntarcticGrid.from_config(config)
    )

    snapshot = {
        "_classification": classification,
        "_sea_ice": {
            "lat": np.asarray(forecast["lat"], dtype=float),
            "lon": np.asarray(forecast["lon"], dtype=float),
            "concentration": np.asarray(forecast["concentration"], dtype=float),
        },
        "_icebergs": dict(iceberg_data),
        "_weather": weather_severity,
        "_weather_ok": weather_ok,
    }
    _SOURCE_CACHE[fingerprint] = (time.time(), snapshot)
    logger.info("ROUTE_DATA_LOADED cache_stored=true elapsed_ms=%.1f", (time.perf_counter() - started) * 1000)
    return snapshot


def warm_route_data(config: dict[str, Any]) -> None:
    """Load the route data snapshot before the first request on a cold worker."""
    started = time.perf_counter()
    _load_source_snapshot(datetime.now(timezone.utc).replace(tzinfo=None), config)
    grid = AntarcticGrid.from_config(config)
    land_mask_path = settings.LAND_MASK_FILE
    if not land_mask_path or not Path(land_mask_path).is_file():
        land_mask_path = str(DEFAULT_LAND_MASK_FILE)
    _LAND_MASK_CACHE[str(land_mask_path)] = load_land_mask(grid, land_mask_path)
    logger.info("ROUTE_DATA_WARMED elapsed_ms=%.1f", (time.perf_counter() - started) * 1000)


def _real_weather_severity(grid: AntarcticGrid) -> tuple[np.ndarray, bool]:
    """Regrid the newest real weather timestep onto the navigation grid.

    Severity is derived from ``wind_speed`` (>= 30 m/s -> 1.0, linear below).
    The newest weather part on disk is used and its coordinates are mapped onto
    the grid cells by nearest neighbour. When no real weather exists the caller
    receives a zero baseline with ``ok=False`` so it can surface an honest
    "neutral baseline" warning instead of synthetic data.
    """
    import pandas as pd
    from scipy.spatial import cKDTree

    weather_root = (
        Path(__file__).resolve().parents[2] / "Real data" / "processed" / "weather" / "csv"
    )
    empty = np.zeros((grid.nlat, grid.nlon), dtype=float)
    if not weather_root.is_dir():
        return empty, False
    parts = sorted(weather_root.glob("weather_part_*.csv"))
    if not parts:
        return empty, False
    path = parts[-1]

    key = path.name
    if key not in _WEATHER_SEVERITY_CACHE:
        try:
            latest_ts = None
            latest_frames: list[pd.DataFrame] = []
            for chunk in pd.read_csv(
                path,
                usecols=["timestamp", "latitude", "longitude", "wind_speed"],
                chunksize=200_000,
            ):
                chunk_ts = chunk["timestamp"].max()
                if latest_ts is None or chunk_ts > latest_ts:
                    latest_ts = chunk_ts
                    latest_frames = [chunk[chunk["timestamp"] == latest_ts]]
                elif chunk_ts == latest_ts:
                    latest_frames.append(chunk[chunk["timestamp"] == latest_ts])
            if not latest_frames:
                return empty, False
            frame = pd.concat(latest_frames, ignore_index=True)
        except Exception:  # noqa: BLE001 - report, never crash routing
            return empty, False
        if frame.empty:
            return empty, False
        frame = frame[["latitude", "longitude", "wind_speed"]]
        if frame.empty:
            return empty, False
        ws = pd.to_numeric(frame["wind_speed"], errors="coerce").to_numpy(dtype=float)
        ws = np.where(np.isfinite(ws), np.clip(ws / 30.0, 0.0, 1.0), 0.0)
        wlat = frame["latitude"].to_numpy(dtype=float)
        wlon = frame["longitude"].to_numpy(dtype=float)

        grid_lat, grid_lon = np.meshgrid(grid.lats, grid.lons, indexing="ij")
        gpoints = np.column_stack([grid_lat.ravel(), grid_lon.ravel()])
        tree = cKDTree(np.column_stack([wlat, wlon]))
        _, idx = tree.query(gpoints, k=1)
        sev = ws[idx].reshape(grid_lat.shape)
        _WEATHER_SEVERITY_CACHE[key] = (sev, True)
    return _WEATHER_SEVERITY_CACHE[key]


def build_route_engine(
    vessel: dict[str, Any],
    ts: datetime | None,
    config: dict[str, Any],
) -> tuple[AntarcticGrid, RiskEngine, list[str], str]:
    """Assemble the Antarctic routing grid + risk engine from configured data.

    Returns ``(grid, engine, notes, data_classification)``. ``ts`` is the
    requested forecast time (UTC); ``None`` means now.
    """
    if ts is None:
        ts = datetime.now(timezone.utc).replace(tzinfo=None)
    elif ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)

    started = time.perf_counter()
    logger.info("ROUTE_CALCULATION_STARTED")
    grid = AntarcticGrid.from_config(config)
    risk_config = RiskConfig.from_dict(config)
    engine = RiskEngine(grid, risk_config)

    horizon = int(config.get("forecast_horizon_hours", 24))
    notes: list[str] = []

    snapshot = _load_source_snapshot(ts, config)
    classification = str(snapshot.get("_classification", "unknown"))

    if "synthetic_demo" in classification:
        notes.append("Sea-ice data is synthetic demo data.")

    s_ice = snapshot["_sea_ice"]
    engine.add_sea_ice(s_ice["lat"], s_ice["lon"], s_ice["concentration"])

    iceberg_data = snapshot["_icebergs"]
    ib_class = str(iceberg_data.get("classification", "unknown"))
    if ib_class == "synthetic_demo":
        notes.append("Iceberg positions are synthetic demo data.")
    elif iceberg_data.get("trained_used"):
        notes.append(
            f"Iceberg predictions use trained model '{iceberg_data.get('model')}' "
            "on real observations."
        )
    else:
        notes.append(
            f"Iceberg positions from real observations "
            f"(baseline model '{iceberg_data.get('model')}')."
        )
    engine.add_icebergs(iceberg_data.get("icebergs", []))

    weather_severity = snapshot["_weather"]
    weather_ok = snapshot["_weather_ok"]
    engine.add_weather_severity(weather_severity)
    if weather_ok:
        notes.append("Weather severity derived from the newest real wind field.")
    else:
        notes.append(
            "No real weather severity available; routing used a neutral baseline "
            "instead of synthetic data."
        )

    engine.set_vessel_constraints(vessel)
    engine.apply_iceberg_hard_block()
    engine.apply_sea_ice_hazard_blocks()

    land_mask_path = settings.LAND_MASK_FILE
    if not land_mask_path or not Path(land_mask_path).is_file():
        land_mask_path = str(DEFAULT_LAND_MASK_FILE)
    land_mask_result = _LAND_MASK_CACHE.get(str(land_mask_path))
    if land_mask_result is None:
        logger.info("LAND_MASK_CACHE_MISS")
        land_mask_result = load_land_mask(grid, land_mask_path)
        _LAND_MASK_CACHE[str(land_mask_path)] = land_mask_result
    land_mask, mask_info = land_mask_result
    if land_mask is not None:
        grid.set_land_mask(land_mask)
        notes.append(
            f"Land mask loaded ({mask_info.get('format', 'unknown')}): "
            f"{mask_info.get('land_cells', 0)} cells marked land."
        )
    else:
        notes.append(mask_info.get("reason", "Land mask not available; open water assumed."))
    logger.info(
        "LAND_MASK_LOADED available=%s land_cells=%d",
        land_mask is not None,
        int(grid.land_mask.sum()),
    )

    engine.compute_total()
    logger.info(
        "ROUTE_CALCULATION_COMPLETED blocked_cells=%d elapsed_ms=%.1f",
        int(engine.blocked_mask.sum()),
        (time.perf_counter() - started) * 1000,
    )
    return grid, engine, notes, classification


def build_debug(
    grid: AntarcticGrid, engine: RiskEngine, config: dict[str, Any]
) -> dict[str, Any]:
    """Debug metadata about the routing grid and hazard state."""
    return {
        "grid": {
            "lat_min": grid.lat_min,
            "lat_max": grid.lat_max,
            "lon_min": grid.lon_min,
            "lon_max": grid.lon_max,
            "resolution_degrees": grid.resolution,
            "nlat": grid.nlat,
            "nlon": grid.nlon,
        },
        "land_cells": int(grid.land_mask.sum()),
        "blocked_cells": int(engine.blocked_mask.sum()),
        "forecast_horizon_hours": int(config.get("forecast_horizon_hours", 24)),
        "risk_weights": dict(engine.config.risk_weights),
        "engine_notes": list(engine.notes),
    }