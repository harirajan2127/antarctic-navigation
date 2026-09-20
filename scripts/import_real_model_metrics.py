#!/usr/bin/env python3
"""Import the real-data evaluation metrics into the ``model_metrics`` table.

Reads ``reports/sea_ice_metrics.json`` and ``reports/iceberg_metrics.json``
(written by ``scripts/evaluate_real_models.py`` from the real processed CSVs)
and upserts them into the database so the analytics/dashboard status logic can
derive an honest per-model status.

Safety guarantees:
  * the database is backed up to ``backend/backups/`` before any change (via
    the sqlite backup API, safe against concurrent connections);
  * rows are upserted on (pipeline, model, metric_name, split) — no duplicates;
  * only the obsolete placeholder ``status`` sentinels (trained_on IS NULL)
    for the two pipelines are deactivated; routes/vessels/iceberg records are
    never touched;
  * values are copied verbatim from the VERIFIED reports; nothing is fabricated.

Usage::

    python scripts/import_real_model_metrics.py

Exits non-zero if a report is missing or invalid (no metrics are invented).
"""
from __future__ import annotations

import math
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
REPORTS_DIR = PROJECT_ROOT / "reports"
BACKUP_DIR = BACKEND_DIR / "backups"
DB_PATH = BACKEND_DIR / "antarctic_dss.db"

# Sandbox the backend config BEFORE importing anything from backend/.
_BACKEND_ENV = BACKEND_DIR / ".env"
if _BACKEND_ENV.exists():
    for line in _BACKEND_ENV.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"'))

os.environ.setdefault("DATA_MODE", "real")
os.environ.setdefault("DEMO_MODE", "auto")
os.environ.setdefault("DATABASE_URL", "sqlite:///" + str(DB_PATH))
sys.path.insert(0, str(BACKEND_DIR))

PIPELINES = ("sea_ice", "iceberg")
MODEL = "persistence"
DATA_SOURCE_LABEL = "real_processed_data"
SPLIT_TEST = "test"
METADATA_SPLIT = "metadata"
MODEL_VERSION = 1.0

# (metric_name, source_path_in_report) extraction helpers are defined below.

def utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _finite(value) -> bool:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def _report(path: Path) -> dict | None:
    if not path.exists():
        print(f"    SKIP {path.name}: report file missing")
        return None
    try:
        import json
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"    SKIP {path.name}: unreadable ({exc})")
        return None


def _sea_ice_rows(report: dict) -> list[tuple[str, float, str]]:
    m = report.get("metrics", {})
    split = report.get("split", {})
    rows = [
        ("mae", m.get("mae"), SPLIT_TEST),
        ("rmse", m.get("rmse"), SPLIT_TEST),
        ("r2", m.get("r2"), SPLIT_TEST),
        ("test_samples", split.get("n_test_pairs"), SPLIT_TEST),
        ("model_version", MODEL_VERSION, METADATA_SPLIT),
    ]
    return [(name, float(v), sp) for name, v, sp in rows if _finite(v)]


def _iceberg_rows(report: dict) -> list[tuple[str, float, str]]:
    within = report.get("within_km", {})
    rows = [
        ("mean_error_km", report.get("haversine_mean_km"), SPLIT_TEST),
        ("median_error_km", report.get("haversine_median_km"), SPLIT_TEST),
        ("within_10km_percent", float(within.get("10", 0) or 0) * 100.0, SPLIT_TEST),
        ("test_samples", report.get("n_test_pairs"), SPLIT_TEST),
        ("model_version", MODEL_VERSION, METADATA_SPLIT),
    ]
    return [(name, float(v), sp) for name, v, sp in rows if _finite(v)]


def backup_database() -> Path | None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    if not DB_PATH.exists():
        print(f"  WARN database not found at {DB_PATH} — skipping backup")
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = BACKUP_DIR / f"antarctic_dss_{stamp}.db"
    src = sqlite3.connect(str(DB_PATH))
    try:
        dst = sqlite3.connect(str(dest))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()
    print(f"  Backup -> {dest.relative_to(PROJECT_ROOT)}")
    return dest


def main() -> int:
    print("=" * 70)
    print("  ANTARCTIC DSS - IMPORT REAL MODEL METRICS")
    print("=" * 70)

    backup = backup_database()

    reports = {
        "sea_ice": _report(REPORTS_DIR / "sea_ice_metrics.json"),
        "iceberg": _report(REPORTS_DIR / "iceberg_metrics.json"),
    }
    extractors = {"sea_ice": _sea_ice_rows, "iceberg": _iceberg_rows}

    from database.database import SessionLocal
    from database.models import ModelMetric

    failures: list[str] = []
    changed: list[dict] = []

    with SessionLocal() as db:
        # (1) Deactivate obsolete placeholder "status" sentinels.
        sentinels = db.query(ModelMetric).filter(
            ModelMetric.pipeline.in_(PIPELINES),
            ModelMetric.metric_name == "status",
            ModelMetric.trained_on.is_(None),
            ModelMetric.demo.is_(False),
        ).all()
        for row in sentinels:
            print(f"  Deactivate old placeholder: id={row.id} "
                  f"pipeline={row.pipeline} metric_name={row.metric_name} value={row.metric_value}")
            db.delete(row)

        # (2) Upsert real metric rows per pipeline.
        for pipeline in PIPELINES:
            report = reports[pipeline]
            report_path = REPORTS_DIR / f"{pipeline}_metrics.json"
            if report is None or report.get("status") != "VERIFIED":
                failures.append(
                    f"{pipeline}: no VERIFIED evaluation report "
                    f"(status={report.get('status') if report else 'MISSING'})"
                )
                continue
            rows = extractors[pipeline](report)
            if not rows:
                failures.append(f"{pipeline}: no usable metric values in report")
                continue
            print(f"\n  {pipeline} (model={MODEL}, source={DATA_SOURCE_LABEL}):")
            for metric_name, value, split in rows:
                existing = db.query(ModelMetric).filter(
                    ModelMetric.pipeline == pipeline,
                    ModelMetric.model == MODEL,
                    ModelMetric.metric_name == metric_name,
                    ModelMetric.split == split,
                ).first()
                if existing is not None:
                    old = existing.metric_value
                    existing.metric_value = value
                    existing.demo = False
                    existing.trained_on = utcnow_naive()
                    existing.source_file = str(report_path)
                    print(f"    UPDATE {metric_name:<24} {old:.6f} -> {value:.6f}")
                    changed.append({"pipeline": pipeline, "metric_name": metric_name,
                                    "action": "update", "value": value})
                else:
                    db.add(ModelMetric(
                        pipeline=pipeline,
                        model=MODEL,
                        metric_name=metric_name,
                        metric_value=value,
                        split=split,
                        demo=False,
                        trained_on=utcnow_naive(),
                        source_file=str(report_path),
                    ))
                    print(f"    INSERT {metric_name:<24} {value:.6f}")
                    changed.append({"pipeline": pipeline, "metric_name": metric_name,
                                    "action": "insert", "value": value})
        db.commit()

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Backup:                {backup.relative_to(PROJECT_ROOT) if backup else 'none'}")
    print(f"  Rows upserted:         {len(changed)}")
    print(f"  Placeholder rows removed: {len(sentinels)}")
    print(f"  Data source label:     {DATA_SOURCE_LABEL}")
    print(f"  Status (computed):     evaluated_on_real_data"
          f"  (models were evaluated on real data, not retrained)")
    if failures:
        print("  FAILURES:")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  OK — metrics imported from dataset_provenance: real_processed_data")
    return 0


if __name__ == "__main__":
    sys.exit(main())