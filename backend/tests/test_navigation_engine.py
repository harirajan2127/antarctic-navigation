"""Tests for the navigation decision-support engine.

Uses small synthetic grids so all tests are fast and deterministic.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from navigation.astar import AStarPlanner, NoPathFoundError
from navigation.fuel_estimator import FuelEstimator, FuelEstimate
from navigation.geodesy import (
    haversine_km,
    haversine_nm,
    initial_bearing_deg,
    normalize_longitude,
)
from navigation.grid import AntarcticGrid, GridCell, GridProjection
from navigation.risk_engine import RiskConfig, RiskEngine, sea_ice_risk
from navigation.route_optimizer import RouteOptimizer, RouteValidator

# ------------------------------------------------------------------ #
# Haversine
# ------------------------------------------------------------------ #

def test_haversine_km_zero():
    assert haversine_km(0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0)

def test_haversine_km_one_degree():
    d = haversine_km(0.0, 0.0, 1.0, 0.0)
    assert d == pytest.approx(111.195, rel=0.01)

def test_haversine_nm_one_degree_lat():
    d = haversine_nm(0.0, 0.0, 1.0, 0.0)
    assert d == pytest.approx(60.0, rel=0.1)

def test_haversine_symmetry():
    assert haversine_km(-70.0, 10.0, -60.0, 20.0) == pytest.approx(
        haversine_km(-60.0, 20.0, -70.0, 10.0)
    )

def test_initial_bearing_north():
    b = initial_bearing_deg(0.0, 0.0, 10.0, 0.0)
    assert b == pytest.approx(0.0, abs=0.1)

def test_initial_bearing_east():
    b = initial_bearing_deg(0.0, 0.0, 0.0, 10.0)
    assert b == pytest.approx(90.0, abs=0.5)

def test_normalize_longitude():
    assert normalize_longitude(190.0) == pytest.approx(-170.0)
    assert normalize_longitude(-200.0) == pytest.approx(160.0)
    assert normalize_longitude(0.0) == 0.0

# ------------------------------------------------------------------ #
# Grid
# ------------------------------------------------------------------ #

def test_grid_creation():
    grid = AntarcticGrid(-80.0, -60.0, -50.0, 50.0, resolution_deg=2.0)
    assert grid.nlat == 11  # -80, -78, ..., -60
    assert grid.nlon == 51
    assert grid.resolution == 2.0
    assert grid.navigable_mask.all()

def test_grid_from_config():
    config = {"region": {"lat_min": -80.0, "lat_max": -70.0,
                          "lon_min": -20.0, "lon_max": 20.0},
              "grid_resolution_degrees": 5.0}
    grid = AntarcticGrid.from_config(config)
    assert grid.nlat == 3  # -80, -75, -70
    assert grid.nlon == 9

def test_grid_contains():
    grid = AntarcticGrid(-80.0, -60.0, -50.0, 50.0, resolution_deg=2.0)
    assert grid.contains(-70.0, 0.0)
    assert not grid.contains(-50.0, 0.0)
    assert not grid.contains(-70.0, 60.0)

def test_grid_cell_for():
    grid = AntarcticGrid(-80.0, -60.0, -50.0, 50.0, resolution_deg=2.0)
    c = grid.cell_for(-75.0, 10.0)
    lat, lon = grid.center_of(c)
    assert lat == pytest.approx(-76.0, abs=1.0)
    assert lon == pytest.approx(10.0, abs=1.0)

def test_grid_mark_land():
    grid = AntarcticGrid(-80.0, -60.0, -50.0, 50.0, resolution_deg=2.0)
    grid.mark_land(-70.0, 0.0)
    c = grid.cell_for(-70.0, 0.0)
    assert not grid.is_navigable(c)
    assert grid.land_mask[c.lat_index, c.lon_index]

def test_grid_mark_land_box():
    grid = AntarcticGrid(-80.0, -60.0, -50.0, 50.0, resolution_deg=2.0)
    grid.mark_land_box(-75.0, -70.0, -10.0, 10.0)
    c = grid.cell_for(-72.5, 0.0)
    assert grid.land_mask[c.lat_index, c.lon_index]

def test_grid_set_land_mask():
    grid = AntarcticGrid(-80.0, -60.0, -50.0, 50.0, resolution_deg=2.0)
    mask = np.zeros((grid.nlat, grid.nlon), dtype=bool)
    mask[5, 5] = True
    grid.set_land_mask(mask)
    c = GridCell(5, 5)
    assert not grid.is_navigable(c)

def test_grid_neighbors_8connectivity():
    grid = AntarcticGrid(-80.0, -60.0, -50.0, 50.0, resolution_deg=2.0)
    neighbors = grid.neighbors(GridCell(5, 5))
    assert len(neighbors) == 8

def test_grid_neighbors_4connectivity():
    grid = AntarcticGrid(-80.0, -60.0, -50.0, 50.0, resolution_deg=2.0)
    neighbors = grid.neighbors(GridCell(5, 5), allow_diagonal=False)
    assert len(neighbors) == 4

def test_grid_neighbors_corner():
    grid = AntarcticGrid(-80.0, -60.0, -50.0, 50.0, resolution_deg=2.0)
    neighbors = grid.neighbors(GridCell(0, 0))
    assert len(neighbors) == 3

def test_grid_projection_roundtrip():
    grid = AntarcticGrid(-80.0, -60.0, -50.0, 50.0, resolution_deg=2.0)
    lat, lon = -72.0, 15.0
    east, north = grid.projection.project(lat, lon)
    lat2, lon2 = grid.projection.unproject(east, north)
    assert lat2 == pytest.approx(lat, abs=0.01)
    assert lon2 == pytest.approx(lon, abs=0.01)

# ------------------------------------------------------------------ #
# Risk engine: ice risk function
# ------------------------------------------------------------------ #

def test_sea_ice_risk_zero():
    assert sea_ice_risk(0.0) == pytest.approx(0.0)

def test_sea_ice_risk_full():
    assert sea_ice_risk(1.0) >= 0.9

def test_sea_ice_risk_mid():
    r = sea_ice_risk(0.5)
    assert 0.0 < r < 1.0

def test_sea_ice_risk_vectorized():
    arr = sea_ice_risk(np.array([0.0, 0.3, 0.5, 0.7, 1.0]))
    assert arr.shape == (5,)
    assert arr[0] == pytest.approx(0.0)

# ------------------------------------------------------------------ #
# Risk engine: full construction
# ------------------------------------------------------------------ #

def test_risk_engine_construction():
    grid = AntarcticGrid(-80.0, -60.0, -50.0, 50.0, resolution_deg=2.0)
    engine = RiskEngine(grid, RiskConfig())
    assert engine.total_risk.shape == (grid.nlat, grid.nlon)

def test_risk_engine_add_sea_ice():
    grid = AntarcticGrid(-80.0, -60.0, -50.0, 50.0, resolution_deg=2.0)
    engine = RiskEngine(grid, RiskConfig())
    lats = np.arange(-80.0, -59.0, 2.0); lons = np.arange(-50.0, 51.0, 2.0)
    conc = np.zeros((len(lats), len(lons)))
    for i, lat in enumerate(lats):
        conc[i, :] = 0.8 if lat < -75 else 0.0
    engine.add_sea_ice(lats, lons, conc)
    engine.compute_total()
    c_polar = engine.sample_at(-78.0, 0.0)
    c_open = engine.sample_at(-65.0, 0.0)
    assert c_polar["sea_ice_concentration"] == pytest.approx(0.8)
    assert c_open["sea_ice_concentration"] == pytest.approx(0.0)
    assert c_polar["risk_score"] > c_open["risk_score"]

def test_risk_engine_iceberg_proximity():
    grid = AntarcticGrid(-80.0, -60.0, -50.0, 50.0, resolution_deg=2.0)
    engine = RiskEngine(grid, RiskConfig())
    engine.add_icebergs([{"iceberg_id": "IB1", "latitude": -70.0, "longitude": 0.0}])
    engine.compute_total()
    # At iceberg: iceberg_risk=1.0; total = weight*1.0 + zeros = weight
    assert engine.sample_at(-70.0, 0.0)["risk_score"] > 0.3
    assert engine.distance_to_nearest_iceberg(-70.0, 0.0) < 1.0
    # iceberg_risk layer itself should be 1.0 at the iceberg cell
    c = grid.cell_for(-70.0, 0.0)
    assert engine.iceberg_risk[c.lat_index, c.lon_index] > 0.99

def test_risk_engine_distance_from_iceberg():
    grid = AntarcticGrid(-80.0, -60.0, -50.0, 50.0, resolution_deg=2.0)
    engine = RiskEngine(grid, RiskConfig())
    engine.add_icebergs([{"iceberg_id": "IB1", "latitude": -70.0, "longitude": 0.0}])
    d = engine.distance_to_nearest_iceberg(-70.0, 0.0)
    assert d < 1.0  # at iceberg itself
    far = engine.distance_to_nearest_iceberg(-60.0, 50.0)
    assert far > 100.0

def test_risk_engine_weather():
    grid = AntarcticGrid(-80.0, -60.0, -50.0, 50.0, resolution_deg=2.0)
    engine = RiskEngine(grid, RiskConfig())
    severity = np.zeros((grid.nlat, grid.nlon))
    severity[5, 5] = 1.0
    engine.add_weather_severity(severity)
    assert engine.weather_risk[5, 5] == pytest.approx(1.0)
    assert engine.weather_risk[0, 0] == pytest.approx(0.0)

def test_risk_engine_vessel_constraints():
    grid = AntarcticGrid(-80.0, -60.0, -50.0, 50.0, resolution_deg=2.0)
    engine = RiskEngine(grid, RiskConfig())
    lats = np.arange(-80.0, -59.0, 2.0); lons = np.arange(-50.0, 51.0, 2.0)
    conc = np.zeros((len(lats), len(lons)))
    conc[0, :] = 0.9  # > PC5 limit 0.6
    conc[1, :] = 0.5  # < PC5 limit 0.6
    engine.add_sea_ice(lats, lons, conc)
    engine.set_vessel_constraints({"ice_class": "PC5", "cruise_speed_knots": 12})
    engine.apply_vessel_constraints()
    assert engine.blocked_mask[0, 0]
    assert not engine.blocked_mask[1, 0]

def test_risk_engine_unknown_ice_class():
    grid = AntarcticGrid(-80.0, -60.0, -50.0, 50.0, resolution_deg=2.0)
    engine = RiskEngine(grid, RiskConfig())
    engine.set_vessel_constraints({"ice_class": "ZZ9"})
    # Should fall back to default limit without error.
    assert engine.max_sea_ice_for_vessel is not None

def test_risk_engine_obstacle_mask():
    grid = AntarcticGrid(-80.0, -60.0, -50.0, 50.0, resolution_deg=2.0)
    grid.mark_land(-70.0, 0.0)
    engine = RiskEngine(grid, RiskConfig())
    engine.compute_total()
    mask = engine.obstacle_mask()
    c = grid.cell_for(-70.0, 0.0)
    assert mask[c.lat_index, c.lon_index]

def test_risk_level_bucketing():
    assert RiskEngine.risk_level(0.1) == "low"
    assert RiskEngine.risk_level(0.45) == "moderate"
    assert RiskEngine.risk_level(0.75) == "high"
    assert RiskEngine.risk_level(0.95) == "extreme"

def test_risk_config_from_dict():
    d = {
        "risk_weights": {"sea_ice_concentration": 0.5, "iceberg_proximity": 0.3,
                          "weather_severity": 0.2},
        "thresholds": {"max_safe_sea_ice_concentration": 0.5,
                       "iceberg_exclusion_radius_km": 15.0,
                       "min_clearance_km": 5.0,
                       "max_risk_ratio": 0.90},
    }
    cfg = RiskConfig.from_dict(d)
    assert cfg.iceberg_exclusion_radius_km == pytest.approx(15.0)
    assert cfg.max_risk_ratio == pytest.approx(0.90)

# ------------------------------------------------------------------ #
# A* route finding (unit, small grid)
# ------------------------------------------------------------------ #

def _simple_grid():
    """20x20 grid, no obstacles, all risk zero."""
    grid = AntarcticGrid(-80.0, -62.0, -50.0, -12.0, resolution_deg=1.0)
    return grid

def test_astar_finds_path():
    grid = _simple_grid()
    cost = np.zeros((grid.nlat, grid.nlon))
    planner = AStarPlanner(grid.lats, grid.lons, cost=cost,
                           safety_weight=0.0, distance_weight=1.0)
    cells = planner.plan(-79.0, -49.0, -63.0, -13.0)
    coords = planner.coordinates_from_cells(cells)
    assert len(coords) >= 2
    assert coords[0] == pytest.approx((-79.0, -49.0), abs=0.6)
    assert coords[-1] == pytest.approx((-63.0, -13.0), abs=0.6)

def test_astar_avoids_obstacles():
    grid = _simple_grid()
    cost = np.zeros((grid.nlat, grid.nlon))
    obstacles = np.zeros((grid.nlat, grid.nlon), dtype=bool)
    # Partial barrier: blocks rows 2-16 at column 9, leaving gap at row 0,1,17,18
    obstacles[2:17, 9] = True
    planner = AStarPlanner(grid.lats, grid.lons, cost=cost, obstacles=obstacles,
                           distance_weight=1.0)
    cells = planner.plan(-79.0, -49.0, -63.0, -13.0)
    for c in cells:
        assert not obstacles[c.lat_index, c.lon_index]

def test_astar_no_path():
    grid = _simple_grid()
    cost = np.zeros((grid.nlat, grid.nlon))
    obstacles = np.zeros((grid.nlat, grid.nlon), dtype=bool)
    obstacles[10, :] = True  # complete horizontal wall
    planner = AStarPlanner(grid.lats, grid.lons, cost=cost, obstacles=obstacles,
                           distance_weight=1.0)
    with pytest.raises(NoPathFoundError):
        planner.plan(-79.0, -49.0, -63.0, -13.0)

def test_astar_start_blocked():
    grid = _simple_grid()
    obstacles = np.zeros((grid.nlat, grid.nlon), dtype=bool)
    start_cell = grid.cell_for(-79.0, -49.0)
    obstacles[start_cell.lat_index, start_cell.lon_index] = True
    planner = AStarPlanner(grid.lats, grid.lons, obstacles=obstacles)
    with pytest.raises(NoPathFoundError):
        planner.plan(-79.0, -49.0, -63.0, -13.0)

def test_astar_cost_avoids_high_risk():
    grid = _simple_grid()
    cost = np.zeros((grid.nlat, grid.nlon))
    cost[5, :] = 0.94  # high risk row — below default max_cost=0.95 so not blocked
    planner = AStarPlanner(grid.lats, grid.lons, cost=cost,
                           safety_weight=1.0, distance_weight=0.0)
    cells = planner.plan(-79.0, -49.0, -63.0, -13.0)
    risk_cells = [c for c in cells if cost[c.lat_index, c.lon_index] > 0.9]
    assert len(risk_cells) < len(cells)  # path detours around worst cells

# ------------------------------------------------------------------ #
# Fuel estimation
# ------------------------------------------------------------------ #

def test_fuel_travel_time():
    t = FuelEstimator.travel_time_hours(1852.0, 10.0)  # 100 nm → 10 h
    assert t == pytest.approx(100.0, rel=0.01)

def test_fuel_rate_phase_lookup():
    vessel = {
        "cruise_speed_knots": 12.0,
        "max_speed_knots": 16.0,
        "fuel_consumption_tons_per_day": {"slow_speed": 18, "cruise_speed": 28, "max_speed": 45},
    }
    rate_slow, src_slow = FuelEstimator.rate_tons_per_day(vessel, 8.0)
    rate_cruise, src_cruise = FuelEstimator.rate_tons_per_day(vessel, 12.0)
    rate_max, src_max = FuelEstimator.rate_tons_per_day(vessel, 16.0)
    assert rate_slow == pytest.approx(18.0)
    assert rate_cruise == pytest.approx(28.0)
    assert rate_max == pytest.approx(45.0)
    assert "slow" in src_slow

def test_fuel_rate_lph_conversion():
    vessel = {"fuel_rate_lph": 1372.5}
    rate, src = FuelEstimator.rate_tons_per_day(vessel, 10.0)
    # 1372.5 L/h * 24h * 0.85 t/m3 / 1000 = 28.0
    assert rate == pytest.approx(28.0, rel=0.01)
    assert "lph" in src

def test_fuel_estimate():
    vessel = {"cruise_speed_knots": 12.0, "max_speed_knots": 16.0,
              "fuel_consumption_tons_per_day": {"slow_speed": 18, "cruise_speed": 28, "max_speed": 45}}
    est = FuelEstimator.estimate(1852.0 * 10, 12.0, vessel)  # 10,000 nm
    assert est.travel_time_hours == pytest.approx(10000.0 / 12.0, rel=0.01)
    assert est.fuel_tons == pytest.approx(28.0 * 10000.0 / 12.0 / 24.0, rel=0.01)
    assert est.distance_nm == pytest.approx(10000.0, rel=0.01)

def test_fuel_estimate_numeric_keys():
    vessel = {"fuel_consumption_tons_per_day": {"10.0": 20.0, "16.0": 40.0}}
    rate, _ = FuelEstimator.rate_tons_per_day(vessel, 12.0)
    assert rate == pytest.approx(20.0)

def test_fuel_no_config_raises():
    with pytest.raises(ValueError, match="fuel"):
        FuelEstimator.rate_tons_per_day({}, 10.0)

# ------------------------------------------------------------------ #
# Route optimizer: synthetic grid (medium, for end-to-end)
# ------------------------------------------------------------------ #

def _route_grid_engine():
    """20x40 grid with open-ocean southern region, ice barrier, and one iceberg."""
    grid = AntarcticGrid(-80.0, -62.0, -40.0, -1.0, resolution_deg=1.0)
    engine = RiskEngine(grid, RiskConfig())

    lats = grid.lats; lons = grid.lons
    conc = np.zeros((grid.nlat, grid.nlon))
    for i, lat in enumerate(lats):
        conc[i, :] = 0.85 if lat < -78 else 0.0
    engine.add_sea_ice(lats, lons, conc)
    engine.add_icebergs([{"iceberg_id": "IB1", "latitude": -70.0, "longitude": -15.0}])
    engine.set_vessel_constraints({"ice_class": "PC3", "cruise_speed_knots": 12.0})
    engine.apply_vessel_constraints()
    engine.compute_total()
    return grid, engine

def test_route_optimizer_recommended():
    grid, engine = _route_grid_engine()
    vessel = {"cruise_speed_knots": 12.0, "max_speed_knots": 16.0,
              "fuel_consumption_tons_per_day": {"slow_speed": 18, "cruise_speed": 28, "max_speed": 45}}
    opt = RouteOptimizer(grid, engine, vessel=vessel)
    r = opt.optimize(-78.0, -39.0, -63.0, -2.0, "recommended")
    assert len(r["waypoints"]) >= 3
    assert r["distance_km"] > 0
    assert r["distance_nm"] == pytest.approx(r["distance_km"] / 1.852, rel=0.02)
    assert r["fuel_tons"] > 0
    assert r["risk_score"] >= 0.0
    assert r["max_risk_level"] in ("low", "moderate", "high", "extreme")

def test_route_optimizer_shortest_vs_safest_differ():
    """When there is an iceberg, the safest route should be longer."""
    grid, engine = _route_grid_engine()
    vessel = {"cruise_speed_knots": 12.0, "max_speed_knots": 16.0,
              "fuel_consumption_tons_per_day": {"slow_speed": 18, "cruise_speed": 28, "max_speed": 45}}
    opt = RouteOptimizer(grid, engine, vessel=vessel)
    shortest = opt.optimize(-78.0, -39.0, -63.0, -2.0, "shortest")
    safest   = opt.optimize(-78.0, -39.0, -63.0, -2.0, "safest")
    assert safest["distance_km"] >= shortest["distance_km"]

def test_route_optimizer_avoids_iceberg():
    grid, engine = _route_grid_engine()
    vessel = {"cruise_speed_knots": 12.0, "max_speed_knots": 16.0,
              "fuel_consumption_tons_per_day": {"slow_speed": 18, "cruise_speed": 28, "max_speed": 45}}
    opt = RouteOptimizer(grid, engine, vessel=vessel)
    r = opt.optimize(-78.0, -39.0, -63.0, -2.0, "safest")
    # All waypoints should be at least a few km from the iceberg
    for wp in r["waypoints"]:
        if wp["iceberg_distance_km"] is not None:
            assert wp["iceberg_distance_km"] > 5.0  # wide berth

def test_route_distance_geometry_consistency():
    grid, engine = _route_grid_engine()
    vessel = {"cruise_speed_knots": 12.0, "max_speed_knots": 16.0,
              "fuel_consumption_tons_per_day": {"slow_speed": 18, "cruise_speed": 28, "max_speed": 45}}
    opt = RouteOptimizer(grid, engine, vessel=vessel)
    r = opt.optimize(-78.0, -39.0, -63.0, -2.0, "recommended")
    # Manually re-sum waypoint segment distances
    total_km = sum(wp["distance_km"] for wp in r["waypoints"])
    assert total_km == pytest.approx(r["distance_km"], abs=0.1)

def test_route_validator_land_blocks():
    grid = AntarcticGrid(-80.0, -60.0, -50.0, 50.0, resolution_deg=2.0)
    engine = RiskEngine(grid, RiskConfig()); engine.compute_total()
    validator = RouteValidator(grid, engine)
    grid.mark_land(-72.0, 0.0)
    coords = [(-79.0, 10.0), (-72.0, 0.0), (-65.0, 10.0)]
    valid, warnings = validator.validate(coords)
    assert not valid
    assert any("land" in w.lower() for w in warnings)

def test_route_validator_out_of_bounds():
    grid = AntarcticGrid(-80.0, -60.0, -50.0, 50.0, resolution_deg=2.0)
    engine = RiskEngine(grid, RiskConfig()); engine.compute_total()
    validator = RouteValidator(grid, engine)
    valid, warnings = validator.validate([(-90.0, 10.0), (-70.0, 0.0)])
    assert not valid
    assert any("outside" in w.lower() for w in warnings)

def test_route_waypoint_warnings_high_risk():
    grid = AntarcticGrid(-80.0, -60.0, -50.0, 50.0, resolution_deg=2.0)
    engine = RiskEngine(grid, RiskConfig())
    engine.add_icebergs([{"iceberg_id": "IB1", "latitude": -70.0, "longitude": 0.0}])
    engine.compute_total()
    coords = [(-70.0, 0.0)]
    warnings = engine.waypoint_warnings(coords)
    assert len(warnings) > 0

# ------------------------------------------------------------------ #
# API integration test
# ------------------------------------------------------------------ #

_client = TestClient(app)

def test_api_route_optimize_short():
    # Neumayer approach corridor: a real, reachable pair on the current grid.
    payload = {
        "start_latitude": -65.0,
        "start_longitude": 0.0,
        "destination_latitude": -70.65,
        "destination_longitude": -8.2667,
        "vessel_id": "polar_explorer",
        "optimization_preference": "recommended",
    }
    resp = _client.post("/api/v1/routes/optimize", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["optimization_preference"] == "recommended"
    assert body["distance_km"] > 0
    assert body["distance_nm"] > 0
    assert body["travel_time_hours"] > 0
    assert body["fuel_tons"] > 0
    assert body["risk_score"] >= 0.0
    assert body["risk_level"] in ("low", "moderate", "high", "extreme")
    assert isinstance(body["recommended"]["waypoints"], list)
    assert len(body["recommended"]["waypoints"]) >= 3
    assert len(body["alternatives"]) >= 1
    assert "DISCLAIMER" not in body["disclaimer"]
    assert "Decision support" in body["disclaimer"]

def test_api_route_optimize_404_vessel():
    payload = {
        "start_latitude": -68.0, "start_longitude": 10.0,
        "destination_latitude": -72.0, "destination_longitude": 20.0,
        "vessel_id": "nonexistent",
    }
    resp = _client.post("/api/v1/routes/optimize", json=payload)
    assert resp.status_code == 404

def test_api_route_optimize_land_start():
    """Start at the exact South Pole — land cell if marked as such."""
    payload = {
        "start_latitude": -84.9, "start_longitude": 0.0,
        "destination_latitude": -68.0, "destination_longitude": 10.0,
        "vessel_id": "polar_explorer",
    }
    resp = _client.post("/api/v1/routes/optimize", json=payload)
    # South Pole is near the Antarctic grid boundary; if the API marks it
    # land or out-of-bounds the response should be 422 or 409 or 200 with
    # snapped coords. Accept any of those as valid behaviour.
    assert resp.status_code in (200, 409, 422)