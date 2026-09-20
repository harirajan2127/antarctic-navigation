"""Regression tests for research-station ocean-approach routing.

Problem: with a real land mask enabled, the demo destination used a placeholder
coordinate (-75.00, 165.00) that sits on land, so route planning failed with
"Destination point lies on land." and the overview / analytics pages showed
errors instead of a route.

Fix contract (unchanged by this regression):
* The land mask stays fully active — no route waypoint crosses Antarctic land.
* Research centers keep their real coordinates as metadata.
* Routing to an approved port / research center that lies on land snaps to the
  nearest navigable ocean cell (dateline-safe) and warns when the approach is
  far offshore.
* Arbitrary non-configured land points are still rejected with a clear error.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import app as modern_app
from app.main import app as v1_app
from navigation.approach import (
    PLACE_MATCH_DISTANCE_KM,
    nearest_navigable_point,
    resolve_endpoint,
)

MCMURDO = {"name": "McMurdo Station", "latitude": -77.8469, "longitude": 166.6687}
NEUMAYER = {"name": "Neumayer Station III", "latitude": -70.6500, "longitude": -8.2667}

PORT = {"name": "Hobart", "port_id": "hobart_au", "latitude": -42.8826, "longitude": 147.3257}


@pytest.fixture(scope="module")
def client():
    with TestClient(modern_app) as c:
        yield c


@pytest.fixture(scope="module")
def v1_client():
    with TestClient(v1_app) as c:
        yield c


@pytest.fixture(scope="module")
def grid_with_mask():
    from app.config import load_navigation_config
    from config import settings
    from navigation.grid import AntarcticGrid
    from navigation.land_mask import load_land_mask

    config = load_navigation_config()
    grid = AntarcticGrid.from_config(config)
    mask, info = load_land_mask(grid, settings.LAND_MASK_FILE)
    assert mask is not None, "Land mask must be loaded for these tests."
    grid.set_land_mask(mask)
    return grid


# --------------------------------------------------------------------------
# Unit tests — navigation/approach.py
# --------------------------------------------------------------------------
def test_nearest_navigable_point_already_ocean(grid_with_mask):
    grid = grid_with_mask
    lat, lon = -68.0, 30.0
    c = grid.cell_for(lat, lon)
    assert grid.is_navigable(c)
    cell_lat = grid.lats[c.lat_index]
    cell_lon = grid.lons[c.lon_index]
    rlat, rlon, dist = nearest_navigable_point(
        grid.lats, grid.lons, grid.navigable_mask, lat, lon
    )
    assert (rlat, rlon) == (cell_lat, cell_lon)
    assert dist == 0.0


def test_nearest_navigable_point_mcmurdo_snaps(grid_with_mask):
    """A land station snaps to the nearest navigable cell in its own sector."""
    grid = grid_with_mask
    cell = grid.cell_for(MCMURDO["latitude"], MCMURDO["longitude"])
    assert not grid.is_navigable(cell)
    rlat, rlon, dist = nearest_navigable_point(
        grid.lats, grid.lons, grid.navigable_mask,
        MCMURDO["latitude"], MCMURDO["longitude"],
    )
    assert dist > 0.0
    assert grid.is_navigable(grid.cell_for(rlat, rlon))
    # Same-sector: longitude stays near the station (no dateline wrap).
    assert 136.0 <= rlon <= 180.0
    assert rlat > MCMURDO["latitude"]  # nearest ocean cell lies north


def test_resolve_endpoint_landsnaps_configured_place(grid_with_mask):
    ep = resolve_endpoint(
        grid_with_mask,
        MCMURDO["latitude"],
        MCMURDO["longitude"],
        [MCMURDO],
        [PORT],
        max_approach_km=100.0,
    )
    assert ep.on_land is True
    assert ep.place_name == MCMURDO["name"]
    assert ep.place_kind == "research_center"
    assert ep.approach_distance_km > 0.0
    assert grid_with_mask.is_navigable(grid_with_mask.cell_for(ep.lat, ep.lon))
    # far-approach warning (distance lies beyond the configured threshold)
    assert "ocean approach" in (ep.note or "").lower()
    assert "exceeds the configured approach distance" in (ep.note or "")


def test_resolve_endpoint_near_approach_warning(grid_with_mask):
    """A short (< threshold) approach uses the plain offshore wording."""
    ep = resolve_endpoint(
        grid_with_mask,
        NEUMAYER["latitude"],
        NEUMAYER["longitude"],
        [NEUMAYER],
        [PORT],
        max_approach_km=200.0,
    )
    assert ep.on_land is True
    assert ep.approach_distance_km < 200.0
    assert "nearest valid ocean approach point" in (ep.note or "")
    assert "exceeds the configured approach distance" not in (ep.note or "")


def test_resolve_endpoint_rejects_unknown_land(grid_with_mask):
    with pytest.raises(ValueError, match="lies on land"):
        resolve_endpoint(
            grid_with_mask,
            -72.0,
            20.0,
            [MCMURDO],
            [PORT],
            max_approach_km=100.0,
        )


def test_resolve_endpoint_openwater_unchanged(grid_with_mask):
    lat, lon = -67.0, 60.0
    assert grid_with_mask.is_navigable(grid_with_mask.cell_for(lat, lon))
    ep = resolve_endpoint(
        grid_with_mask, lat, lon, [MCMURDO], [PORT], max_approach_km=100.0
    )
    assert ep.on_land is False
    assert ep.place_name is None
    assert ep.approach_distance_km == 0.0


# --------------------------------------------------------------------------
# API integration — POST /api/routes/optimize  (modern endpoint)
# --------------------------------------------------------------------------
def _assert_station_route_details(body):
    assert body["destination_name"] == MCMURDO["name"]
    assert body["requested_destination_latitude"] == pytest.approx(MCMURDO["latitude"], abs=0.001)
    assert body["requested_destination_longitude"] == pytest.approx(MCMURDO["longitude"], abs=0.001)
    assert body["ocean_approach_distance_km"] is not None
    assert body["ocean_approach_distance_km"] > 0.0
    rec = body["recommended"]
    assert len(rec["waypoints"]) > 2
    assert rec["distance_km"] > 0
    assert any("ocean approach" in w.lower() for w in rec["warnings"])


def test_modern_routes_mcmurdo_no_longer_lies_on_land(client):
    resp = client.post(
        "/api/routes/optimize",
        json={
            "start_latitude": -65.0,
            "start_longitude": 140.0,
            "destination_latitude": MCMURDO["latitude"],
            "destination_longitude": MCMURDO["longitude"],
            "vessel_id": "polar_explorer",
            "preference": "recommended",
        },
    )
    assert resp.status_code == 200, resp.text
    _assert_station_route_details(resp.json())


def test_modern_routes_approach_point_is_not_land(client, grid_with_mask):
    body = client.post(
        "/api/routes/optimize",
        json={
            "start_latitude": -65.0,
            "start_longitude": 140.0,
            "destination_latitude": MCMURDO["latitude"],
            "destination_longitude": MCMURDO["longitude"],
            "vessel_id": "polar_explorer",
            "preference": "recommended",
        },
    ).json()
    ap_cell = grid_with_mask.cell_for(
        body["destination_latitude"], body["destination_longitude"]
    )
    assert grid_with_mask.is_navigable(ap_cell)
    for wp in body["recommended"]["waypoints"]:
        cell = grid_with_mask.cell_for(wp["lat"], wp["lon"])
        assert grid_with_mask.is_navigable(cell), f"waypoint {wp} crosses land"


def test_modern_routes_neumayer_reachable(client):
    resp = client.post(
        "/api/routes/optimize",
        json={
            "start_latitude": -65.0,
            "start_longitude": 0.0,
            "destination_latitude": NEUMAYER["latitude"],
            "destination_longitude": NEUMAYER["longitude"],
            "vessel_id": "polar_explorer",
            "preference": "recommended",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["destination_name"] == NEUMAYER["name"]
    assert body["ocean_approach_distance_km"] is not None
    assert body["ocean_approach_distance_km"] > 0.0
    assert any("ocean approach" in w.lower() for w in body["recommended"]["warnings"])


def test_modern_routes_unknown_land_rejected(client):
    resp = client.post(
        "/api/routes/optimize",
        json={
            "start_latitude": -65.0,
            "start_longitude": 140.0,
            "destination_latitude": -72.0,
            "destination_longitude": 20.0,
            "vessel_id": "polar_explorer",
            "preference": "recommended",
        },
    )
    assert resp.status_code in (409, 422)
    assert "lies on land" in resp.json()["detail"].lower()


# --------------------------------------------------------------------------
# API integration — POST /api/v1/routes/optimize  (legacy v1 endpoint)
# --------------------------------------------------------------------------
def test_v1_routes_mcmurdo_resolves(v1_client):
    resp = v1_client.post(
        "/api/v1/routes/optimize",
        json={
            "start_latitude": -65.0,
            "start_longitude": 140.0,
            "destination_latitude": MCMURDO["latitude"],
            "destination_longitude": MCMURDO["longitude"],
            "vessel_id": "polar_explorer",
            "optimization_preference": "recommended",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["destination_name"] == MCMURDO["name"]
    assert body["requested_destination_latitude"] == pytest.approx(MCMURDO["latitude"], abs=0.001)
    assert body["ocean_approach_distance_km"] is not None
    assert body["ocean_approach_distance_km"] > 0.0
    assert any("ocean approach" in w.lower() for w in body["warnings"])


def test_v1_routes_neumayer_reachable(v1_client):
    resp = v1_client.post(
        "/api/v1/routes/optimize",
        json={
            "start_latitude": -65.0,
            "start_longitude": 0.0,
            "destination_latitude": NEUMAYER["latitude"],
            "destination_longitude": NEUMAYER["longitude"],
            "vessel_id": "polar_explorer",
            "optimization_preference": "recommended",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["destination_name"] == NEUMAYER["name"]
    assert body["ocean_approach_distance_km"] is not None
    assert body["ocean_approach_distance_km"] > 0.0


def test_v1_routes_unknown_land_rejected(v1_client):
    resp = v1_client.post(
        "/api/v1/routes/optimize",
        json={
            "start_latitude": -65.0,
            "start_longitude": 140.0,
            "destination_latitude": -72.0,
            "destination_longitude": 20.0,
            "vessel_id": "polar_explorer",
            "optimization_preference": "recommended",
        },
    )
    assert resp.status_code == 422
    assert "lies on land" in resp.json()["detail"].lower()