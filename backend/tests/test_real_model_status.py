"""Validation tests for the real-data model-status fix.

These tests confirm that after ``scripts/import_real_model_metrics.py`` has run
the analytics/dashboard no longer reports "Demo Only" for the persistence
pipelines that were evaluated on real processed data.

Every assertion reads values from the database / reports — nothing is
hardcoded. The tests require the import to have been executed; if the DB is
missing the real metric rows they fail deliberately.
"""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from config import settings
from database.database import SessionLocal
from database.models import ModelMetric
from main import app

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BACKEND_DIR.parent / "reports"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _db_metrics(pipeline: str) -> list[ModelMetric]:
    with SessionLocal() as db:
        return list(
            db.scalars(
                select(ModelMetric)
                .where(ModelMetric.pipeline == pipeline, ModelMetric.demo == False)  # noqa: E712
                .order_by(ModelMetric.metric_name)
            ).all()
        )


def _report(pipeline: str) -> dict:
    path = REPORTS_DIR / f"{pipeline}_metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def summary(client):
    resp = client.get("/api/analytics/summary")
    assert resp.status_code == 200
    return resp.json()


# --------------------------------------------------------------------------
# 1-2. Real metrics exist for both pipelines
# --------------------------------------------------------------------------
def test_sea_ice_real_metrics_exist():
    rows = _db_metrics("sea_ice")
    assert rows, "no sea_ice metric rows with demo=False in model_metrics"
    names = {r.metric_name for r in rows}
    assert {"mae", "rmse", "r2", "test_samples"} <= names


def test_iceberg_real_metrics_exist():
    rows = _db_metrics("iceberg")
    assert rows, "no iceberg metric rows with demo=False in model_metrics"
    names = {r.metric_name for r in rows}
    assert {"mean_error_km", "median_error_km", "within_10km_percent", "test_samples"} <= names


# --------------------------------------------------------------------------
# 3-4. Status is not "Demo Only"
# --------------------------------------------------------------------------
def test_sea_ice_status_not_demo(summary):
    status = summary["model_status"]["sea_ice"]["status"]
    assert status in ("trained_on_real_data", "evaluated_on_real_data"), f"unexpected: {status}"


def test_iceberg_status_not_demo(summary):
    status = summary["model_status"]["iceberg"]["status"]
    assert status in ("trained_on_real_data", "evaluated_on_real_data"), f"unexpected: {status}"


# --------------------------------------------------------------------------
# 5. Real-data source is stored / derived correctly
# --------------------------------------------------------------------------
def test_real_data_source_stored():
    for pipeline in ("sea_ice", "iceberg"):
        rows = _db_metrics(pipeline)
        assert all(r.demo is False for r in rows), f"{pipeline}: a demo=True row slipped in"
        evid = {r.source_file for r in rows if r.metric_name != "model_version"}
        assert evid and all(p and p.endswith(f"{pipeline}_metrics.json") for p in evid), (
            f"{pipeline}: source_file should point at the real evaluation report"
        )
    s = _summary_from_db()
    for pipeline in ("sea_ice", "iceberg"):
        assert s[pipeline]["training_data_source"] == "real_processed_data", pipeline


def _summary_from_db():
    import sys
    sys.path.insert(0, str(BACKEND_DIR))
    from services.analytics_service import _model_status_info
    return _model_status_info()


# --------------------------------------------------------------------------
# 6. Model names match across DB, API, frontend labels
# --------------------------------------------------------------------------
def test_model_names_consistent(summary):
    for pipeline, display in (("sea_ice", "Sea-Ice"), ("iceberg", "Iceberg")):
        entry = summary["model_status"][pipeline]
        assert entry["model_name"] == pipeline
        assert entry["display_name"] == display
        rows = _db_metrics(pipeline)
        assert all(r.pipeline == pipeline for r in rows)
        assert all(r.model == "persistence" for r in rows)


# --------------------------------------------------------------------------
# 7. Metrics are not hardcoded (report -> DB -> API chain)
# --------------------------------------------------------------------------
def test_metrics_not_hardcoded(summary):
    for pipeline in ("sea_ice", "iceberg"):
        api_metrics = summary["model_status"][pipeline]["metrics"]
        db_values = {r.metric_name: r.metric_value for r in _db_metrics(pipeline)
                     if r.metric_name != "model_version"}
        report = _report(pipeline)
        assert db_values, f"{pipeline}: no DB metrics to compare"
        assert api_metrics, f"{pipeline}: API exposed no metrics"
        for name, value in db_values.items():
            assert name in api_metrics and api_metrics[name] == pytest.approx(value), (
                f"{pipeline}: API metric {name} does not match DB row"
            )
        if report.get("status") == "VERIFIED":
            if pipeline == "sea_ice":
                assert api_metrics["mae"] == pytest.approx(report["metrics"]["mae"])
                assert api_metrics["test_samples"] == pytest.approx(
                    report["split"]["n_test_pairs"])
            else:
                assert api_metrics["mean_error_km"] == pytest.approx(
                    report["haversine_mean_km"])
                assert api_metrics["test_samples"] == pytest.approx(report["n_test_pairs"])


# --------------------------------------------------------------------------
# 8. No demo metric rows are returned in REAL DATA mode
# --------------------------------------------------------------------------
def test_no_demo_records_in_real_mode():
    if not settings.data_mode_real:
        pytest.skip("not running in REAL data mode")
    with SessionLocal() as db:
        demo_rows = db.scalars(select(ModelMetric).where(ModelMetric.demo == True)).all()  # noqa: E712
    assert not demo_rows, f"{len(demo_rows)} demo metric rows leaked into real mode"


# --------------------------------------------------------------------------
# 9. API returns the correct real metric records
# --------------------------------------------------------------------------
def test_api_returns_real_metric_records(summary):
    assert summary["model_status"]["sea_ice"]["metrics_available"] is True
    assert summary["model_status"]["iceberg"]["metrics_available"] is True
    si = summary["model_status"]["sea_ice"]["metrics"]
    ib = summary["model_status"]["iceberg"]["metrics"]
    assert si["r2"] > 0.8 and si["mae"] < 0.1 and si["test_samples"] > 1_000_000
    assert 20 < ib["mean_error_km"] < 80 and ib["test_samples"] >= 100
    assert summary["real_model_available"] == {"sea_ice": True, "iceberg": True}


# --------------------------------------------------------------------------
# 10. The misleading warning is gone
# --------------------------------------------------------------------------
def test_no_misleading_warning(summary):
    assert "No trained real-data model metrics found in the database." not in summary["warnings"]