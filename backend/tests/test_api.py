"""API-level tests using the FastAPI TestClient."""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"]


def test_list_ports():
    resp = client.get("/api/v1/config/ports")
    assert resp.status_code == 200
    ports = resp.json()
    assert len(ports) >= 5
    names = {p["name"] for p in ports}
    assert "Hobart" in names
    assert "Cape Town" in names


def test_list_research_centers():
    resp = client.get("/api/v1/config/research-centers")
    assert resp.status_code == 200
    centers = resp.json()
    assert len(centers) >= 5
    all_ids = {c["center_id"] for c in centers}
    assert "bharati_in" in all_ids


def test_list_vessels():
    resp = client.get("/api/v1/vessels/")
    assert resp.status_code == 200
    vessels = resp.json()
    assert len(vessels) >= 1


def test_sea_ice_forecast():
    resp = client.get("/api/v1/sea-ice/forecast", params={"horizon_hours": 24})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "persistence"
    assert body["horizon_hours"] == 24


def test_iceberg_predictions():
    resp = client.get("/api/v1/icebergs/predictions", params={"horizon_hours": 24})
    assert resp.status_code == 200
    body = resp.json()
    assert "icebergs" in body
    assert len(body["icebergs"]) > 0


def test_start_outbound_journey():
    payload = {
        "departure_port_id": "hobart_au",
        "destination_id": "casey_au",
        "journey_mode": "outbound",
        "position_mode": "simulation",
    }
    resp = client.post("/api/v1/navigation/journey", json=payload)
    assert resp.status_code == 200
    route = resp.json()
    assert route["journey_mode"] == "outbound"
    assert route["total_distance_nm"] > 1000
    assert len(route["waypoints"]) > 2


def test_two_hour_rolling_update():
    payload = {
        "departure_port_id": "hobart_au",
        "destination_id": "casey_au",
        "journey_mode": "outbound",
    }
    created = client.post("/api/v1/navigation/journey", json=payload).json()
    journey_id = created["journey_id"]

    # 20 consecutive two-hour updates.
    for _ in range(20):
        resp = client.post(f"/api/v1/navigation/journey/{journey_id}/advance")
        assert resp.status_code == 200
        route = resp.json()
        assert route["route_update_count"] >= 1

    status = client.get(f"/api/v1/navigation/journey/{journey_id}/status")
    assert status.status_code == 200
    assert status.json()["journey_id"] == journey_id
    # The vessel should have advanced, not reset to origin.
    assert status.json()["current_time_hours"] > 0


def test_return_journey_reverses_origin_destination():
    payload = {
        "departure_port_id": "cape_town_za",
        "destination_id": "bharati_in",
        "journey_mode": "return",
    }
    resp = client.post("/api/v1/navigation/journey", json=payload)
    assert resp.status_code == 200
    route = resp.json()
    assert route["journey_mode"] == "return"
    assert route["origin"] == "Bharati Station"
    assert route["destination"] == "Cape Town"