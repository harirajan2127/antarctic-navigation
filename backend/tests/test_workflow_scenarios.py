"""End-to-end workflow scenario tests (legacy /api/v1 journey lifecycle).

Covers the accepted scenarios:

1. Hobart -> research center (outbound)
2. Cape Town -> research center (outbound)
3. Ushuaia -> research center (outbound)
4. Research center -> departure port (return)
5. Two-hour rolling navigation: start, route, advance 2 h, position change,
   route recalculation from the current position, repeat, and arrival.

These tests drive the *same* HTTP surface as the live backend
(``app.main:app`` mounted at ``/api/v1``) so they double as the demo workflow
contract.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.utils.geo import haversine_distance_nm

client = TestClient(app)


def _start(port_id: str, center_id: str, journey_mode: str = "outbound"):
    resp = client.post(
        "/api/v1/navigation/journey",
        json={
            "departure_port_id": port_id,
            "destination_id": center_id,
            "journey_mode": journey_mode,
            "position_mode": "simulation",
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


# ------------------------------------------------------------------ #
# Scenarios 1-3: outbound journeys
# ------------------------------------------------------------------ #

@pytest.mark.parametrize(
    "port_id,center_id,port_name,center_name,min_distance_nm",
    [
        ("hobart_au", "mcmurdo_us", "Hobart", "McMurdo Station", 1200),   # Scenario 1
        ("cape_town_za", "bharati_in", "Cape Town", "Bharati Station", 1500),  # Scenario 2
        ("ushuaia_ar", "mcmurdo_us", "Ushuaia", "McMurdo Station", 3000),  # Scenario 3
    ],
)
def test_outbound_journey_from_each_port(
    port_id, center_id, port_name, center_name, min_distance_nm
):
    route = _start(port_id, center_id)
    assert route["journey_mode"] == "outbound"
    assert route["origin"] == port_name
    assert route["destination"] == center_name
    assert route["total_distance_nm"] > min_distance_nm
    assert route["estimated_duration_hours"] > 0
    assert route["route_update_count"] == 1
    assert len(route["waypoints"]) >= 3
    assert route["max_risk_level"] in ("low", "moderate", "high", "extreme")
    # Vessel starts at the departure port.
    p_lat, p_lon = _port_coords(port_id)
    assert haversine_distance_nm(
        route["current_vessel_lat"], route["current_vessel_lon"], p_lat, p_lon
    ) < 1.0
    # Every waypoint carries a decision-support risk level.
    assert {wp["risk_level"] for wp in route["waypoints"]} <= {
        "low", "moderate", "high", "extreme",
    }


def _port_coords(port_id: str) -> tuple[float, float]:
    from app.config import load_ports

    for p in load_ports():
        if p["port_id"] == port_id:
            return float(p["latitude"]), float(p["longitude"])
    raise KeyError(port_id)


# ------------------------------------------------------------------ #
# Scenario 4: return journey (research center -> departure port)
# ------------------------------------------------------------------ #

@pytest.mark.parametrize(
    "center_id,port_id,center_name,port_name",
    [
        ("mcmurdo_us", "hobart_au", "McMurdo Station", "Hobart"),
        ("bharati_in", "cape_town_za", "Bharati Station", "Cape Town"),
    ],
)
def test_return_journey_reverses_origin_and_destination(
    center_id, port_id, center_name, port_name
):
    route = _start(port_id, center_id, journey_mode="return")
    assert route["journey_mode"] == "return"
    # Origin/destination are swapped for the inbound leg. On the legacy grid
    # (lat -85..-55) route distance is measured inside the grid domain.
    assert route["origin"] == center_name
    assert route["destination"] == port_name
    assert route["total_distance_nm"] > 1200
    assert len(route["waypoints"]) >= 3


# ------------------------------------------------------------------ #
# Scenario 5: two-hour rolling navigation
# ------------------------------------------------------------------ #

def test_two_hour_rolling_recalculates_from_current_position():
    route = _start("hobart_au", "casey_au")
    journey_id = route["journey_id"]
    start_lat, start_lon = route["current_vessel_lat"], route["current_vessel_lon"]
    assert route["route_update_count"] == 1

    prev_lat, prev_lon = start_lat, start_lon
    prev_count = 1

    for step in range(1, 4):
        resp = client.post(f"/api/v1/navigation/journey/{journey_id}/advance")
        assert resp.status_code == 200, resp.text
        updated = resp.json()

        # 1+2. Position must have changed after advancing 2 simulated hours.
        assert (updated["current_vessel_lat"], updated["current_vessel_lon"]) != (
            prev_lat, prev_lon
        ), "vessel position did not advance"

        # 3. Route must have been recalculated from the *current* position:
        #    the update count grows and the new route begins where the vessel is.
        assert updated["route_update_count"] > prev_count
        first_wp = updated["waypoints"][0]
        assert haversine_distance_nm(
            first_wp["latitude"], first_wp["longitude"],
            updated["current_vessel_lat"], updated["current_vessel_lon"],
        ) < 60.0, "recalculated route does not start at the current vessel position"

        prev_lat, prev_lon = updated["current_vessel_lat"], updated["current_vessel_lon"]
        prev_count = updated["route_update_count"]

    status = client.get(f"/api/v1/navigation/journey/{journey_id}/status").json()
    assert status["journey_id"] == journey_id
    # 4. Time accumulates and progress is measured from the port.
    assert status["current_time_hours"] > 0
    assert 0.0 < status["progress_percent"] < 100.0
    assert status["route_update_count"] == prev_count


def test_manual_recalculation_preserves_exact_coordinates():
    route = _start("hobart_au", "casey_au")
    journey_id = route["journey_id"]
    requested_lat, requested_lon = -65.1234, 140.2345

    resp = client.post(
        f"/api/v1/navigation/journey/{journey_id}/recalculate",
        params={"lat": requested_lat, "lon": requested_lon},
    )

    assert resp.status_code == 200, resp.text
    recalculated = resp.json()
    first_wp = recalculated["waypoints"][0]
    assert first_wp["latitude"] == requested_lat
    assert first_wp["longitude"] == requested_lon
    assert recalculated["current_vessel_lat"] == requested_lat
    assert recalculated["current_vessel_lon"] == requested_lon


def test_manual_recalculation_applies_requested_direction():
    route = _start("hobart_au", "casey_au")
    journey_id = route["journey_id"]
    requested_lat, requested_lon = -65.1234, 140.2345

    resp = client.post(
        f"/api/v1/navigation/journey/{journey_id}/recalculate",
        params={"lat": requested_lat, "lon": requested_lon, "direction": "NE"},
    )

    assert resp.status_code == 200, resp.text
    first, second = resp.json()["waypoints"][:2]
    assert (first["latitude"], first["longitude"]) == (requested_lat, requested_lon)
    assert second["latitude"] > requested_lat
    assert second["longitude"] > requested_lon


def test_two_hour_rolling_route_changes_and_reaches_destination():
    """Advance far enough for the recomputed route geometry to change,
    then force arrival at the goal and verify completion."""
    route = _start("hobart_au", "casey_au")
    journey_id = route["journey_id"]

    distances = [route["total_distance_nm"]]
    prev_lat, prev_lon = route["current_vessel_lat"], route["current_vessel_lon"]

    for _ in range(14):
        resp = client.post(f"/api/v1/navigation/journey/{journey_id}/advance")
        assert resp.status_code == 200
        updated = resp.json()
        distances.append(updated["total_distance_nm"])
        assert (updated["current_vessel_lat"], updated["current_vessel_lon"]) != (
            prev_lat, prev_lon
        )
        prev_lat, prev_lon = updated["current_vessel_lat"], updated["current_vessel_lon"]

    # 5. The route genuinely changed as the vessel progressed (different start
    #    point on a live risk grid -> recomputed geometry).
    assert len(set(distances)) >= 2, "route never changed across rolling updates"

    # Force the vessel to the goal and advance once more to complete.
    goal_lat = _center_coords("casey_au")[0]
    goal_lon = _center_coords("casey_au")[1]
    resp = client.post(
        f"/api/v1/navigation/journey/{journey_id}/recalculate",
        params={"lat": goal_lat, "lon": goal_lon},
    )
    assert resp.status_code == 200, resp.text
    resp = client.post(f"/api/v1/navigation/journey/{journey_id}/advance")
    assert resp.status_code == 200
    updated = resp.json()

    status = client.get(f"/api/v1/navigation/journey/{journey_id}/status").json()
    # The legacy grid snaps the destination to a 0.5-degree cell centre, so a
    # small bounded residual (< 30 nm) after arrival is expected behaviour.
    assert status["is_complete"] is True
    assert status["remaining_distance_nm"] < 30.0
    assert 0.0 <= status["progress_percent"] <= 100.0


def _center_coords(center_id: str) -> tuple[float, float]:
    from app.config import load_research_centers

    for c in load_research_centers():
        if c["center_id"] == center_id:
            return float(c["latitude"]), float(c["longitude"])
    raise KeyError(center_id)


# ------------------------------------------------------------------ #
# Journey lifecycle edge behaviour
# ------------------------------------------------------------------ #

def test_journey_404_unknown_id():
    resp = client.post("/api/v1/navigation/journey/does-not-exist/advance")
    assert resp.status_code == 404


def test_journey_404_unknown_port():
    resp = client.post(
        "/api/v1/navigation/journey",
        json={"departure_port_id": "atlantis", "destination_id": "mcmurdo_us"},
    )
    assert resp.status_code == 404


def test_live_mode_requires_gps_position():
    resp = client.post(
        "/api/v1/navigation/journey",
        json={
            "departure_port_id": "hobart_au",
            "destination_id": "mcmurdo_us",
            "position_mode": "live",
        },
    )
    assert resp.status_code == 400