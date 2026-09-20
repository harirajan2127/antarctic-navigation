"""Dataset inventory and status for the DSS REST API.

Answers "what data is available, and is it real or demo?" honestly: every
dataset is classified from the actual loader output (or NetCDF attributes),
never guessed from a filename alone. The classification is also persisted to
the ``dataset_metadata`` table for analytics.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select, update

from app.config import load_datasets
from app.services.iceberg.data_loader import IcebergDataLoader
from app.services.sea_ice.data_loader import SeaIceDataLoader

from config import PIPELINE_DATA, SYNTHETIC_DEMO, settings, effective_demo
from database.database import SessionLocal
from database.models import DatasetRecord
from services.data_paths import (
    OCEAN_NETCDF,
    SEA_ICE_NETCDF,
    WEATHER_NETCDF,
    ICEBERG_CSV,
)

logger = logging.getLogger("dss.services.datasets")

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BACKEND_DIR / "datasets" / "processed"

WEATHER_FILE = WEATHER_NETCDF
OCEAN_FILE = OCEAN_NETCDF

KNOWN_DATASETS = ("sea_ice", "iceberg", "weather", "ocean_currents")

REAL_LABELS = {
    "sea_ice": "real_sea_ice",
    "iceberg": "real_iceberg",
    "ocean_currents": "real_ocean",
    "weather": "real_weather",
    "vessel": "real_vessel",
    "bathymetry": "real_bathymetry",
}


def _display_classification(dataset: str, classification: str) -> str:
    """Map internal pipeline labels to the visible real-data source names.

    ``pipeline_data``/``raw`` mean "processed real observations"; the dashboard
    shows the specific source name (real_sea_ice, real_iceberg, ...) instead.
    """
    if classification in (PIPELINE_DATA, "raw"):
        return REAL_LABELS.get(dataset, classification)
    return classification


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _netcdf_index(path: Path) -> tuple[str | None, str | None, str | None]:
    """Return (classification, source, note) from NetCDF global attrs."""
    if not path.exists():
        return None, None, "file missing"
    try:
        import netCDF4 as nc  # type: ignore

        with nc.Dataset(str(path)) as ds:
            classification = str(getattr(ds, "classification", ""))
            source = str(getattr(ds, "source_name", "") or getattr(ds, "source", ""))
        if classification:
            return classification or None, source or None, None
        return None, source or None, "no classification attribute in file"
    except Exception as exc:  # pragma: no cover - defensive
        return None, None, f"could not inspect NetCDF: {exc}"


def _classify_sea_ice() -> dict:
    try:
        data = SeaIceDataLoader(netcdf_path=str(SEA_ICE_NETCDF)).load()
    except Exception as exc:
        return {"classification": "unclassified", "demo": True, "records": None,
                "file_path": None, "notes": f"loader failed: {exc}"}
    classification = data.get("classification", "unknown")
    return {
        "classification": str(classification),
        "demo": effective_demo(str(classification)),
        "records": None,
        "file_path": data.get("path"),
        "notes": data.get("demo_notice"),
    }


def _classify_iceberg() -> dict:
    try:
        items = IcebergDataLoader(data_path=str(ICEBERG_CSV)).load()
    except Exception as exc:
        return {"classification": "unclassified", "demo": True, "records": None,
                "file_path": None, "notes": f"loader failed: {exc}"}
    classification = items[0].get("classification", "unknown") if items else "unknown"
    path = Path(ICEBERG_CSV) if Path(ICEBERG_CSV).is_file() else None
    return {
        "classification": str(classification),
        "demo": effective_demo(str(classification)),
        "records": len(items) if items else None,
        "file_path": str(path) if path else None,
        "notes": items[0].get("demo_notice") if items else None,
    }


def _csv_classify(path: Path) -> dict:
    """Classify weather/ocean/sea_ice from a resolved real processed CSV path."""
    if not path.exists():
        return {
            "classification": "unavailable" if settings.data_mode_real else "unclassified",
            "demo": not settings.data_mode_real,
            "records": None,
            "file_path": None,
            "notes": "Data Unavailable: no real processed CSV found" if settings.data_mode_real
                     else "file missing",
        }
    return {
        "classification": PIPELINE_DATA,
        "demo": False,
        "records": None,
        "file_path": str(path),
        "notes": None,
    }


def _classify_weather() -> dict:
    if WEATHER_FILE.suffix.lower() in (".csv", ".json"):
        return _csv_classify(WEATHER_FILE)
    classification, source, note = _netcdf_index(WEATHER_FILE)
    return {
        "classification": classification or "unclassified",
        "demo": not classification or classification == SYNTHETIC_DEMO,
        "records": None,
        "file_path": str(WEATHER_FILE) if WEATHER_FILE.exists() else None,
        "notes": note or source,
    }


def _classify_ocean() -> dict:
    if OCEAN_FILE.suffix.lower() in (".csv", ".json"):
        return _csv_classify(OCEAN_FILE)
    classification, source, note = _netcdf_index(OCEAN_FILE)
    return {
        "classification": classification or "unclassified",
        "demo": not classification or classification == SYNTHETIC_DEMO,
        "records": None,
        "file_path": str(OCEAN_FILE) if OCEAN_FILE.exists() else None,
        "notes": note or source,
    }


_CLASSIFIERS = {
    "sea_ice": _classify_sea_ice,
    "iceberg": _classify_iceberg,
    "weather": _classify_weather,
    "ocean_currents": _classify_ocean,
}


def _description(dataset: str) -> str:
    blocks = load_datasets().get(dataset, {}).get("description")
    if isinstance(blocks, str):
        return blocks
    return "Dataset configuration for the Antarctic Navigation DSS"


def _source_meta(dataset: str) -> tuple[str | None, str | None]:
    meta = load_datasets().get(dataset, {})
    sources = meta.get("sources") or []
    if not sources:
        return None, None
    first = sources[0]
    return first.get("source_id"), first.get("name")


def inventory() -> list[dict]:
    """Classify every known dataset (in-memory, honest)."""
    result = []
    for dataset in KNOWN_DATASETS:
        info = _CLASSIFIERS[dataset]()
        source_id, source_name = _source_meta(dataset)
        file_path = info.get("file_path")
        classification = _display_classification(dataset, info.get("classification", "unclassified"))
        if classification == SYNTHETIC_DEMO:
            status = "demo"
        elif classification in ("", "unknown", "unclassified"):
            status = "missing"
        else:
            status = "ok"
        result.append(
            {
                "dataset": dataset,
                "description": _description(dataset),
                "classification": classification,
                "demo": info["demo"],
                "status": status,
                "file_path": file_path,
                "records": info.get("records"),
                "notes": info.get("notes"),
                "source_id": source_id,
                "source_name": source_name,
                "updated_at": _now(),
            }
        )
    return result


def real_data_available(classifications: list[dict] | None = None) -> bool:
    """True when any registered dataset carries real (non-demo) data.

    Classifications that are neither synthetic demo nor unknown/unclassified
    count as real (e.g. ``pipeline_data``, ``real_observation``, ``imported``).
    """
    items = classifications if classifications is not None else list(inventory())
    for item in items:
        if hasattr(item, "classification"):
            cls = str(item.classification)
            demo = item.demo
        else:
            cls = str(item.get("classification") or "unclassified")
            demo = _is_demo(item.get("demo"))
        if cls not in ("", "unknown", "unclassified", "synthetic_demo"):
            return True
        if not _is_demo(demo):
            return True
    return False


def _is_demo(demo: bool | None) -> bool:
    return True if demo is None else bool(demo)


def sync_database() -> None:
    """Persist dataset metadata to the ``dataset_metadata`` table (idempotent)."""
    with SessionLocal() as db:
        for item in inventory():
            stmt = (
                update(DatasetRecord)
                .where(DatasetRecord.dataset_name == item["dataset"])
                .values(
                    classification=item["classification"],
                    status=item["status"],
                    file_path=item["file_path"],
                    records=item["records"],
                    demo=item["demo"],
                    notes=item["notes"],
                    source_id=item["source_id"],
                    source_name=item["source_name"],
                    updated_at=_now(),
                )
            )
            db.execute(stmt)
            if db.scalar(
                select(DatasetRecord.id).where(
                    DatasetRecord.dataset_name == item["dataset"]
                )
            ) is None:
                payload = {k: v for k, v in item.items() if k not in ("dataset", "description")}
                db.add(DatasetRecord(dataset_name=item["dataset"], **payload))
        db.commit()


def from_database() -> list[dict]:
    """Dataset metadata read back from the database, if present."""
    with SessionLocal() as db:
        rows = db.scalars(
            select(DatasetRecord).order_by(DatasetRecord.dataset_name)
        ).all()
    # mirror schemas.DatasetInfo field names
    keymap = {}
    for row in rows:
        keymap[row.dataset_name] = {
            "dataset": row.dataset_name,
            "description": None,
            "classification": row.classification,
            "demo": row.demo,
            "status": row.status,
            "file_path": row.file_path,
            "records": row.records,
            "notes": row.notes,
            "source_id": row.source_id,
            "source_name": row.source_name,
            "updated_at": row.updated_at,
        }
    if keymap:
        return [keymap[name] for name in KNOWN_DATASETS if name in keymap]
    return inventory()