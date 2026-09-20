"""API-level tests for the DSS REST API (``backend/main.py``).

Covers the full ``/api/*`` surface: health, datasets, sea-ice, icebergs
(list/detail/predict/trajectory/distance), routes (optimize/detail), vessels, and
analytics. Uses a module-scoped TestClient so the application lifespan
(DB init + seeding) runs exactly once.
"""
import pytest
from fastapi.testclient import TestClient

from main import app

ROUTE_PAYLOAD = {
    "start_latitude": -65.0,
    "start_longitude": 140.0,
    "destination_latitude": -77.8469,
    "destination_longitude": 166.6687,
    "vessel_id": "polar_explorer",
    "preference": "recommended",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------
def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"]
    assert body["version"]
    assert body["database"] in ("connected", "unavailable")


def test_unknown_path_404(client):
    resp = client.get("/api/does-not-exist")
    assert resp.status_code == 404


def test_openapi_available(client):
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    assert "IcebergsListResponse" in resp.json().get("components", {}).get("schemas", {})


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------
def test_datasets_list(client):
    resp = client.get("/api/datasets")
    assert resp.status_code == 200
    body = resp.json()
    names = {d["dataset"] for d in body["datasets"]}
    assert {"sea_ice", "iceberg", "weather", "ocean_currents"} <= names


def test_datasets_classification_honest(client):
    resp = client.get("/api/datasets/status")
    assert resp.status_code == 200
    body = resp.json()
    for d in body["datasets"]:
        # demo output is always flagged; demo_mode True means everything is
        # synthetic demo data and the response must say so.
        assert d["demo"] == (d["classification"] == "synthetic_demo")
    if body["demo_mode"]:
        assert body["warning"]


def test_datasets_response_never_leaks_secrets(client):
    resp = client.get("/api/datasets")
    text = resp.text.lower()
    assert "password" not in text
    assert "api_key" not in text
    # The dataset list serializes everything, so this is a good global guard.


# --------------------------------------------------------------------------
# Sea ice
# --------------------------------------------------------------------------
def test_sea_ice_current(client):
    resp = client.get("/api/sea-ice/current")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["lat"]) > 0
    assert len(body["lon"]) > 0
    assert len(body["concentration"]) == len(body["lat"])
    if body["demo"]:
        assert body["warning"]


def test_sea_ice_forecast(client):
    resp = client.get("/api/sea-ice/forecast", params={"horizon_hours": 24})
    assert resp.status_code == 200
    body = resp.json()
    assert body["horizon_hours"] == 24
    assert body["model"] == "persistence"
    assert body["mean_concentration"] >= 0.0
    assert body["model_used_real"] is True  # real DATA_MODE => real persistence forecast


def test_sea_ice_forecast_bad_horizon(client):
    resp = client.get("/api/sea-ice/forecast", params={"horizon_hours": 2000})
    assert resp.status_code == 422


def test_sea_ice_predict(client):
    resp = client.post(
        "/api/sea-ice/predict",
        json={"horizon_hours": 48, "model": "persistence"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["horizon_hours"] == 48
    assert body["valid_time"] is not None


# --------------------------------------------------------------------------
# Icebergs
# --------------------------------------------------------------------------
def test_icebergs_list(client):
    resp = client.get("/api/icebergs")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] > 0
    assert body["icebergs"][0]["iceberg_id"]
    if body["demo"]:
        assert body["warning"]


def test_iceberg_detail_and_404(client):
    listing = client.get("/api/icebergs").json()
    iceberg_id = listing["icebergs"][0]["iceberg_id"]
    resp = client.get(f"/api/icebergs/{iceberg_id}")
    assert resp.status_code == 200
    assert resp.json()["iceberg_id"] == iceberg_id
    assert resp.json()["observation_count"] >= 1
    assert client.get("/api/icebergs/NOPE").status_code == 404


def test_iceberg_predict(client):
    listing = client.get("/api/icebergs").json()
    iceberg_id = listing["icebergs"][0]["iceberg_id"]
    resp = client.post(
        "/api/icebergs/predict",
        json={"iceberg_ids": [iceberg_id], "horizon_hours": 24},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["horizon_hours"] == 24
    assert len(body["icebergs"]) == 1
    pred = body["icebergs"][0]
    assert pred["iceberg_id"] == iceberg_id
    assert isinstance(pred["predicted_lat"], float)
    if body["demo"]:
        assert body["warning"]


def test_iceberg_trajectory_and_404(client):
    listing = client.get("/api/icebergs").json()
    iceberg_id = listing["icebergs"][0]["iceberg_id"]
    resp = client.get(f"/api/icebergs/{iceberg_id}/trajectory")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count_observations"] >= 1
    assert body["observations"][0]["step"] == "observation"
    assert client.get("/api/icebergs/NOPE/trajectory").status_code == 404


def test_iceberg_distance(client):
    listing = client.get("/api/icebergs").json()
    ids = [i["iceberg_id"] for i in listing["icebergs"]][:2]
    resp = client.get(
        "/api/icebergs/distance",
        params={"iceberg_a": ids[0], "iceberg_b": ids[1]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["distance_km"] > 0
    assert body["distance_nm"] > 0
    assert client.get(
        "/api/icebergs/distance",
        params={"iceberg_a": ids[0], "iceberg_b": "MISSING"},
    ).status_code == 404


# --------------------------------------------------------------------------
# Routes / navigation
# --------------------------------------------------------------------------
def test_routes_optimize(client):
    resp = client.post("/api/routes/optimize", json=ROUTE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["route_id"]
    rec = body["recommended"]
    assert len(rec["waypoints"]) > 2
    assert rec["distance_km"] > 0
    assert rec["travel_time_hours"] > 0
    assert rec["risk_level"] in ("low", "moderate", "high", "extreme")
    assert len(body["alternatives"]) > 0
    if body["demo"]:
        assert body["disclaimer"]


def test_routes_stored_detail(client):
    created = client.post("/api/routes/optimize", json=ROUTE_PAYLOAD).json()
    resp = client.get(f"/api/routes/{created['route_id']}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["route_id"] == created["route_id"]
    assert body["waypoints_total"] == len(created["recommended"]["waypoints"])
    assert client.get("/api/routes/NOPE").status_code == 404


def test_routes_unknown_vessel(client):
    payload = {**ROUTE_PAYLOAD, "vessel_id": "no-such-vessel"}
    resp = client.post("/api/routes/optimize", json=payload)
    assert resp.status_code == 404


def test_routes_out_of_region_snaps(client):
    payload = {**ROUTE_PAYLOAD, "start_latitude": -45.0}  # north of grid
    resp = client.post("/api/routes/optimize", json=payload)
    assert resp.status_code == 200
    # Snapped start reported in the response body, not the request value.
    assert resp.json()["start_latitude"] <= -55.0


# --------------------------------------------------------------------------
# Analytics
# --------------------------------------------------------------------------
def test_analytics_summary(client):
    resp = client.get("/api/analytics/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert "real_model_available" in body
    assert set(body["real_model_available"]) == {"sea_ice", "iceberg"}
    assert body["icebergs_tracked"] > 0
    assert isinstance(body["routes_stored"], int)


def test_analytics_model_metrics(client):
    resp = client.get("/api/analytics/model-metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] > 0
    for m in body["metrics"]:
        assert m["pipeline"] in ("sea_ice", "iceberg")
        assert isinstance(m["metric_value"], float)
    if body["demo_mode"]:
        assert body["warning"]


# --------------------------------------------------------------------------
# Vessels
# --------------------------------------------------------------------------
def test_vessels_list(client):
    resp = client.get("/api/vessels")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] > 0
    ids = [v["vessel_id"] for v in body["vessels"]]
    assert "polar_explorer" in ids
    for v in body["vessels"]:
        assert v["vessel_id"]
        assert isinstance(v["name"], str)
        assert v["vessel_type"]


# --------------------------------------------------------------------------
# AI Assistant
# --------------------------------------------------------------------------
def test_assistant_chat_basic(client):
    resp = client.post(
        "/api/assistant/chat",
        json={"question": "What is the predicted sea-ice concentration?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert isinstance(body["answer"], str)
    src_names = {s["name"] for s in body["sources"]}
    assert "sea_ice_forecast" in src_names
    assert "datasets_status" in src_names
    assert body["demo_mode"] in (True, False)
    assert "provider" in body["llm"] and "model" in body["llm"]
    assert "configured" in body["llm"]
    # never expose credentials
    assert "api_key" not in str(body["llm"]).lower()
    assert "sk-" not in str(body["llm"]).lower()


def test_assistant_chat_distance_and_ids(client):
    resp = client.post(
        "/api/assistant/chat",
        json={"question": "What is the distance between DEMO-B000 and DEMO-B001?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    names = {s["name"] for s in body["sources"]}
    assert "iceberg_distance" in names
    assert "DEMO-B000" in body["answer"] and "DEMO-B001" in body["answer"]
    assert "km" in body["answer"]


def test_assistant_chat_history_and_horizon(client):
    resp = client.post(
        "/api/assistant/chat",
        json={
            "question": "Where will iceberg DEMO-B000 move in the next 24 hours?",
            "history": [{"role": "user", "content": "Which icebergs are tracked?"}],
            "horizon_hours": 72,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    names = {s["name"] for s in body["sources"]}
    assert "iceberg_trajectory" in names
    assert "DEMO-B000" in body["answer"]


def test_assistant_chat_empty_question_422(client):
    resp = client.post(
        "/api/assistant/chat",
        json={"question": "   "},
    )
    assert resp.status_code == 422


def test_assistant_chat_no_key_leak_in_warnings(client):
    resp = client.post(
        "/api/assistant/chat",
        json={"question": "What are the risks near the selected route?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["warnings"], list)
    joined = " ".join(body["warnings"]).lower()
    assert "api_key" not in joined and "sk-" not in joined