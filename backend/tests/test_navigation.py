"""Tests for geospatial utilities and navigation components."""
import math

import pytest

from app.services.navigation.astar import AStarPlanner
from app.services.navigation.risk_grid import NavigationRiskGrid
from app.utils.geo import haversine_distance_nm, initial_bearing, intermediate_point


def test_haversine_zero():
    assert haversine_distance_nm(0, 0, 0, 0) == pytest.approx(0.0)


def test_haversine_known_distance():
    # Roughly 60 nautical miles per degree of latitude.
    d = haversine_distance_nm(0, 0, 1, 0)
    assert d == pytest.approx(60.0, rel=0.1)


def test_initial_bearing_east():
    # Moving east from equator to zero-latitude stays at 90 degrees initial bearing.
    b = initial_bearing(0, 0, 0, 10)
    assert b == pytest.approx(90.0, abs=0.5)


def test_intermediate_point_halfway():
    p = intermediate_point(0, 0, 0, 10, 0.5)
    assert p.longitude == pytest.approx(5.0, abs=0.1)
    assert p.latitude == pytest.approx(0.0, abs=0.1)


def test_risk_grid_ice_risk_function():
    assert NavigationRiskGrid._ice_risk(0.0) == 0.0
    assert NavigationRiskGrid._ice_risk(1.0) >= 0.9
    assert 0.0 <= NavigationRiskGrid._ice_risk(0.5) <= 1.0


def test_a_star_finds_path():
    import numpy as np

    lats = np.arange(-70.0, -60.0, 0.5)
    lons = np.arange(0.0, 15.0, 0.5)
    risk = np.zeros((len(lats), len(lons)))
    for i in range(len(lats)):
        for j in range(len(lons)):
            risk[i, j] = 0.01 if j in (5, 6) else 0.001  # small cost bump in middle

    planner = AStarPlanner(lats, lons, risk)
    path = planner.plan(-69.0, 1.0, -61.0, 14.0)
    assert len(path) >= 2
    coords = planner.coordinates_from_cells(path)
    assert coords[0] == pytest.approx((-69.0, 1.0))
    assert coords[-1] == pytest.approx((-61.0, 14.0))


def test_navigation_risk_grid_build():
    import numpy as np

    grid = NavigationRiskGrid(lat_min=-80, lat_max=-60, lon_min=-60, lon_max=60, resolution=2.0)
    lats = np.arange(-80, -59, 2.0)
    lons = np.arange(-60, 61, 2.0)
    conc = np.zeros((len(lats), len(lons)))
    for i in range(len(lats)):
        if lats[i] < -75:
            conc[i, :] = 1.0
    grid.add_sea_ice(lats.tolist(), lons.tolist(), conc.tolist())
    grid.add_icebergs([
        {"iceberg_id": "x", "latitude": -70.0, "longitude": 0.0}
    ])
    total = grid.compute_total()
    assert total.shape == grid.total_risk.shape
    assert 0.0 <= total.min() <= total.max() <= 1.0