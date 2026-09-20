"""Analytics service for the DSS REST API.

Aggregates dataset classifications, model metrics (from trained-artifact
``metrics.json`` files, persisted to the DB at startup), route stats, and
iceberg counts into two simple GET endpoints.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select, func

from config import PIPELINE_DATA, SYNTHETIC_DEMO, effective_demo, settings
from database.database import SessionLocal
from database.models import (
    DatasetRecord,
    Iceberg,
    ModelMetric,
    RouteRecord,
    SeaIceForecastRecord,
)

logger = logging.getLogger("dss.services.analytics")

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"
SEA_ICE_METRICS_FILE = BACKEND_DIR / "models" / "sea_ice" / "run" / "metrics.json"
ICEBERG_RUN_FILE = BACKEND_DIR / "models" / "iceberg" / "run_real" / "run.json"
ICEBERG_EVAL_FILE = BACKEND_DIR / "models" / "iceberg" / "run_real" / "evaluation" / "metrics.json"
if not ICEBERG_RUN_FILE.exists():
    ICEBERG_RUN_FILE = BACKEND_DIR / "models" / "iceberg" / "run" / "run.json"
if not ICEBERG_EVAL_FILE.exists():
    ICEBERG_EVAL_FILE = BACKEND_DIR / "models" / "iceberg" / "run" / "evaluation" / "metrics.json"

# ---------------------------------------------------------------------------
# Canonical model identifiers (single source of truth for names).
#   pipeline  -> internal stable id stored in model_metrics / API
#   display   -> visible dashboard label (unchanged)
#   deployed  -> the production model algorithm used at request time
# ---------------------------------------------------------------------------
MODEL_PIPELINES = ("sea_ice", "iceberg")
MODEL_DISPLAY_NAMES = {"sea_ice": "Sea-Ice", "iceberg": "Iceberg"}
MODEL_DEPLOYED = {"sea_ice": "persistence", "iceberg": "random_forest"}

# Training-data source label (used only when evidence confirms real processed data).
TRAINING_DATA_SOURCE_REAL = "real_processed_data"
TRAINING_DATA_SOURCE_DEMO = "demo_synthetic"

# Model status values (derived from DB model_metrics + registry artifacts).
STATUS_TRAINED_REAL = "trained_on_real_data"
STATUS_EVALUATED_REAL = "evaluated_on_real_data"
STATUS_SOURCE_UNKNOWN = "model_available_source_unknown"
STATUS_DEMO = "demo_only"
STATUS_NOT_AVAILABLE = "not_available"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _read_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _insert_metric(db, pipeline: str, model: str, name: str, value: float,
                   split: str | None, demo: bool, trained_on: datetime | None,
                   source_file: str | None) -> None:
    if trained_on is not None and trained_on.tzinfo is not None:
        trained_on = trained_on.replace(tzinfo=None)
    existing = db.scalar(
        select(ModelMetric).where(
            ModelMetric.pipeline == pipeline,
            ModelMetric.model == model,
            ModelMetric.metric_name == name,
            ModelMetric.split == split,
        )
    )
    if existing is not None:
        # Refresh on retrain so the dashboard always reflects the artifacts.
        existing.metric_value = float(value)
        existing.demo = demo
        existing.trained_on = trained_on
        existing.source_file = source_file
        return
    db.add(
        ModelMetric(
            pipeline=pipeline,
            model=model,
            metric_name=name,
            metric_value=float(value),
            split=split,
            demo=demo,
            trained_on=trained_on,
            source_file=source_file,
        )
    )


SEA_ICE_METRIC_NAMES = ("mae", "rmse", "spatial_correlation")
ICEBERG_TRAIN_METRICS = ("best_val_rmse_km", "best_val_mae_km")
ICEBERG_EVAL_METRICS = ("lat_rmse_deg", "lon_rmse_deg", "position_error_km_mean",
                        "position_error_km_median", "position_error_nm_mean")


def _clear_metrics(db) -> None:
    """Drop existing model-metric rows so the table mirrors artifacts exactly.

    Metrics are purely derived from the trained-artifact files on disk, so a
    clear-and-reseed on every startup keeps the dashboard in sync with retrains
    and never mixes stale splits/models into the report.
    """
    for row in db.scalars(select(ModelMetric)).all():
        db.delete(row)


def seed_model_metrics_from_artifacts() -> None:
    """Sync the model-metrics table from trained-artifact files (idempotent)."""
    from config import settings
    with SessionLocal() as db:
        if settings.data_mode_real:
            # In real mode, remove only demo metrics from the table and do NOT
            # re-seed from on-disk demo-only artifacts.  Real/sentinel metrics
            # (demo=False) survive so the dashboard reflects honest status.
            demo_rows = db.scalars(select(ModelMetric).where(ModelMetric.demo == True)).all()
            for row in demo_rows:
                db.delete(row)
            real_run = _read_json(ICEBERG_RUN_FILE)
            if isinstance(real_run, dict) and real_run.get("demo_only") is False:
                test_metrics = real_run.get("metrics", {}).get("test", {})
                for metric_name, value in (
                    ("mean_error_km", test_metrics.get("mean_distance_km")),
                    ("median_error_km", test_metrics.get("median_distance_km")),
                    ("test_samples", test_metrics.get("samples")),
                ):
                    if value is not None:
                        _insert_metric(
                            db,
                            pipeline="iceberg",
                            model=real_run.get("model", "random_forest"),
                            name=metric_name,
                            value=value,
                            split="test",
                            demo=False,
                            trained_on=_file_mtime(ICEBERG_RUN_FILE),
                            source_file=str(ICEBERG_RUN_FILE),
                        )
            if demo_rows or isinstance(real_run, dict):
                db.commit()
            return

        _clear_metrics(db)
        db.flush()
        # --- Sea-ice metrics (models/sea_ice/run/metrics.json) ----------
        # The result blocks are scored on the held-out TEST split.
        si_data = _read_json(SEA_ICE_METRICS_FILE)
        if isinstance(si_data, dict):
            demo_si = bool(si_data.get("demo_only", True))
            trained_on_si = _parse_ts(str(si_data.get("created_utc", "")))
            results = si_data.get("results", {})
            for model_key in ("persistence", "model_b", "model_c"):
                block = results.get(model_key)
                if not isinstance(block, dict):
                    continue
                model_name = block.get("model", model_key)
                for mname in SEA_ICE_METRIC_NAMES:
                    if mname in block:
                        _insert_metric(
                            db,
                            pipeline="sea_ice",
                            model=model_name,
                            name=mname,
                            value=block[mname],
                            split="test",
                            demo=demo_si,
                            trained_on=trained_on_si or _file_mtime(SEA_ICE_METRICS_FILE),
                            source_file=str(SEA_ICE_METRICS_FILE),
                        )
        # --- Iceberg metrics (models/iceberg/run/run.json) ---------------
        ib_data = _read_json(ICEBERG_RUN_FILE)
        if isinstance(ib_data, dict):
            demo_ib = bool(ib_data.get("demo_only", True))
            metrics_ib = ib_data.get("metrics", {})
            train_block = metrics_ib.get("train", {})
            if isinstance(train_block, dict):
                for mname in ICEBERG_TRAIN_METRICS:
                    if mname in train_block:
                        _insert_metric(
                            db,
                            pipeline="iceberg",
                            model=ib_data.get("model", "unknown"),
                            name=mname,
                            value=train_block[mname],
                            split="val",
                            demo=demo_ib,
                            trained_on=_file_mtime(ICEBERG_RUN_FILE),
                            source_file=str(ICEBERG_RUN_FILE),
                        )
        # --- Iceberg eval metrics (models/iceberg/run/evaluation/) -------
        # The eval file's top level is {data, models, per_horizon_bucket, ...};
        # only the ``models`` entry holds real per-model metric rows (scored on
        # held-out test tracks).
        eval_data = _read_json(ICEBERG_EVAL_FILE)
        if isinstance(eval_data, dict):
            demo_ib = bool(eval_data.get("demo_only", True))
            models_block = eval_data.get("models", {})
            if isinstance(models_block, dict):
                for model_name, block in models_block.items():
                    if not isinstance(block, dict):
                        continue
                    for mname in ICEBERG_EVAL_METRICS:
                        if mname in block:
                            _insert_metric(
                                db,
                                pipeline="iceberg",
                                model=model_name,
                                name=mname,
                                value=block[mname],
                                split="test",
                                demo=demo_ib,
                                trained_on=_file_mtime(ICEBERG_EVAL_FILE),
                                source_file=str(ICEBERG_EVAL_FILE),
                            )
        db.commit()


def _parse_ts(val: str) -> datetime | None:
    if not val or val == "None":
        return None
    try:
        from datetime import timezone as _tz
        ts = datetime.fromisoformat(val.replace("Z", "+00:00"))
        return ts.replace(tzinfo=_tz.utc).replace(tzinfo=None)
    except Exception:
        return None


def _file_mtime(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(tzinfo=None)
    except Exception:
        return None


def model_metrics() -> dict:
    """Read persisted model metrics."""
    with SessionLocal() as db:
        rows = db.scalars(select(ModelMetric).order_by(
            ModelMetric.pipeline, ModelMetric.model, ModelMetric.metric_name
        )).all()
    items = [
        {
            "pipeline": r.pipeline,
            "model": r.model,
            "metric_name": r.metric_name,
            "metric_value": r.metric_value,
            "split": r.split,
            "demo": r.demo,
            "trained_on": r.trained_on,
            "source_file": r.source_file,
        }
        for r in rows
    ]
    return {
        "metrics": items,
        "count": len(items),
        "demo_mode": False,
        "warning": None,
    }


def _evaluation_accuracy() -> dict:
    """Read the real-data accuracy reports written by evaluate_real_models.py.

    Metrics are only surfaced when the report status is VERIFIED — otherwise the
    dashboard displays an honest "Accuracy Not Available" label instead of any
    invented number.
    """
    out: dict = {}
    for name, fname in (("sea_ice", "sea_ice_metrics.json"),
                        ("iceberg", "iceberg_metrics.json")):
        data = _read_json(REPORTS_DIR / fname)
        if not isinstance(data, dict) or data.get("status") != "VERIFIED":
            out[name] = {
                "available": False,
                "status": data.get("status") if isinstance(data, dict) else None,
                "reason": "Accuracy Not Available — Ground Truth Missing",
            }
            continue
        item: dict = {
            "available": True,
            "status": data.get("status"),
            "model": data.get("model"),
            "n_test_pairs": (
                data.get("split", {}).get("n_test_pairs")
                if name == "sea_ice" else data.get("n_test_pairs")
            ),
        }
        if name == "sea_ice":
            m = data.get("metrics", {})
            item["target"] = data.get("target")
            item["unit"] = data.get("unit")
            item["horizon_hours"] = data.get("test_horizon_hours")
            item["mae"] = m.get("mae")
            item["rmse"] = m.get("rmse")
            item["r2"] = m.get("r2")
            item["vs_zero_baseline"] = data.get("beats_zero_baseline")
        else:
            item["mean_km"] = data.get("haversine_mean_km")
            item["median_km"] = data.get("haversine_median_km")
            within_10 = data.get("within_km", {}).get("10")
            item["within_10km_pct"] = round(100 * float(within_10), 2) if within_10 is not None else None
        out[name] = item
    return out


def _evaluation_checks() -> dict:
    """Return real navigation checks without converting unavailable to zero."""
    data = _read_json(REPORTS_DIR / "real_model_accuracy_results.json")
    if not isinstance(data, dict):
        return {}
    report_data = data.get("data", data)
    return {
        "two_hour_recalculation": report_data.get("two_hour", {
            "status": "UNAVAILABLE",
            "message": "2-hour ground-truth data unavailable",
        }),
        "land_mask": report_data.get("land_mask", {
            "status": "UNAVAILABLE",
            "message": "Land-mask data unavailable",
        }),
    }


def _accuracy_summary() -> dict:
    """Build the same validated final/provisional score used by the CLI."""
    from services.accuracy_scoring import WEIGHTS, calculate_weighted_accuracy

    reports = {
        name: _read_json(REPORTS_DIR / filename)
        for name, filename in (
            ("sea_ice", "sea_ice_metrics.json"),
            ("iceberg", "iceberg_metrics.json"),
        )
    }
    checks = _evaluation_checks()
    sea_ice = reports["sea_ice"] if isinstance(reports["sea_ice"], dict) else {}
    iceberg = reports["iceberg"] if isinstance(reports["iceberg"], dict) else {}
    r2 = sea_ice.get("metrics", {}).get("r2")
    mean_km = iceberg.get("haversine_mean_km")
    recalculation = checks.get("two_hour_recalculation", {})
    land_mask = checks.get("land_mask", {})
    route_report = _read_json(REPORTS_DIR / "route_engine_evaluation.json")
    components = {
        "sea_ice": float(r2) * 100 if r2 is not None else None,
        "iceberg": (1 - float(mean_km) / 100) * 100 if mean_km is not None else None,
        "route": 100.0 if isinstance(route_report, dict) and route_report.get("route", {}).get("status") == "VERIFIED" else None,
        "recalc": recalculation.get("accuracy_percent") if recalculation.get("status") == "VERIFIED" else None,
        "land_mask": land_mask.get("accuracy_percent") if land_mask.get("status") == "VERIFIED" else None,
    }
    statuses = {
        "recalc": "Insufficient data" if recalculation.get("status") == "INSUFFICIENT_DATA" else "Error" if recalculation.get("status") == "FAILED" else "N/A",
        "land_mask": "Insufficient data" if land_mask.get("status") == "INSUFFICIENT_DATA" else "Error" if land_mask.get("status") == "FAILED" else "N/A",
    }
    result = calculate_weighted_accuracy(components, weights=WEIGHTS, component_status=statuses)
    if result["score"] is not None:
        result["grade"] = "PROVISIONAL" if result["provisional"] else (
            "A (Production-Ready)" if result["score"] >= 85 else
            "B (Research-Grade)" if result["score"] >= 70 else
            "C (Functional)" if result["score"] >= 50 else "D (Experimental)"
        )
    else:
        result["grade"] = "Not Available"
    return result


def _model_status_info() -> dict:
    """Derive honest per-pipeline model status from DB + on-disk artifacts.

    Priority order (a valid *real* metric row always wins over demo artifacts):
      1. a real-trained model file exists            -> trained_on_real_data
      2. a model file exists, source unknown         -> model_available_source_unknown
      3. real (demo=False) metrics exist in DB       -> evaluated_on_real_data
      4. a model file exists, trained on demo data   -> demo_only
      5. nothing valid                              -> not_available

    A model that was only *evaluated* (not retrained) is reported as
    ``evaluated_on_real_data`` — never as ``trained_on_real_data``.
    """
    from services.model_registry import model_registry

    reg = model_registry()
    by_pipeline: dict[str, list[dict]] = {}
    for m in reg.get("models", []):
        by_pipeline.setdefault(m.get("pipeline"), []).append(m)

    out: dict = {}
    with SessionLocal() as db:
        for pipeline in MODEL_PIPELINES:
            real_rows = db.scalars(
                select(ModelMetric)
                .where(ModelMetric.pipeline == pipeline, ModelMetric.demo == False)  # noqa: E712
                .order_by(ModelMetric.metric_name)
            ).all()
            has_real_metrics = len(real_rows) > 0
            metrics = {r.metric_name: r.metric_value for r in real_rows}

            trained = [m for m in by_pipeline.get(pipeline, []) if m.get("available")]
            file_exists = len(trained) > 0
            demo_flags = [m.get("trained_on_demo") for m in trained]

            if file_exists and any(f is False for f in demo_flags):
                status, source = STATUS_TRAINED_REAL, TRAINING_DATA_SOURCE_REAL
            elif file_exists and any(f is None for f in demo_flags):
                status, source = STATUS_SOURCE_UNKNOWN, None
            elif has_real_metrics:
                status, source = STATUS_EVALUATED_REAL, TRAINING_DATA_SOURCE_REAL
            elif file_exists and all(f is True for f in demo_flags):
                status, source = STATUS_DEMO, TRAINING_DATA_SOURCE_DEMO
            else:
                status, source = STATUS_NOT_AVAILABLE, None

            out[pipeline] = {
                "model_name": pipeline,
                "display_name": MODEL_DISPLAY_NAMES[pipeline],
                "deployed_model": MODEL_DEPLOYED[pipeline],
                "status": status,
                "training_data_source": source,
                "metrics_available": has_real_metrics,
                "metrics": metrics,
                "trained_artifacts": [
                    {"name": m.get("name"), "trained_on_demo": m.get("trained_on_demo")}
                    for m in trained
                ],
            }
    return out


def summary(classifications: list[dict] | None = None) -> dict:
    """Build a high-level summary for the analytics/dashboard."""
    from services.dataset_service import real_data_available

    with SessionLocal() as db:
        n_routes = db.scalar(select(func.count()).select_from(RouteRecord)) or 0
        n_icebergs = db.scalar(select(func.count()).select_from(Iceberg)) or 0
        n_forecasts = db.scalar(select(func.count()).select_from(SeaIceForecastRecord)) or 0
        n_metrics = db.scalar(select(func.count()).select_from(ModelMetric)) or 0
        n_real_metric_rows = db.scalar(
            select(func.count()).select_from(ModelMetric)
            .where(ModelMetric.demo == False)  # noqa: E712
        ) or 0
        n_trained = db.scalar(
            select(func.count()).select_from(ModelMetric)
            .where(ModelMetric.demo == False, ModelMetric.trained_on.isnot(None))
        ) or 0

    model_status = _model_status_info()
    # A model counts as "real" when it is trained or evaluated on real data.
    real_models = {
        p: info.get("status") in (STATUS_TRAINED_REAL, STATUS_EVALUATED_REAL)
        for p, info in model_status.items()
    }
    warnings: list[str] = []
    if n_real_metric_rows == 0 and n_metrics > 0:
        warnings.append("No trained real-data model metrics found in the database.")
    if n_routes == 0:
        warnings.append("No routes have been stored yet.")
    if n_icebergs == 0:
        warnings.append("No icebergs seeded in the database.")

    return {
        "demo_mode": False,
        "real_model_available": real_models,
        "model_status": model_status,
        "model_accuracy": _evaluation_accuracy(),
        "evaluation_checks": _evaluation_checks(),
        "accuracy_summary": _accuracy_summary(),
        "real_data_available": real_data_available(classifications),
        "datasets": classifications or [],
        "routes_stored": int(n_routes),
        "icebergs_tracked": int(n_icebergs),
        "sea_ice_forecasts_stored": int(n_forecasts),
        "metrics_available": n_metrics > 0,
        "model_metrics_count": int(n_metrics),
        "warnings": warnings,
    }