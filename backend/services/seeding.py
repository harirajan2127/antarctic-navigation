"""Startup seeding helpers (idempotent database bootstrap).

Populates the ``icebergs`` / ``iceberg_observations`` and ``vessels`` tables
from the data loaders and fleet catalogue so analytics and detail endpoints
have DB-backed counts. All inserts are guarded by primary-key existence and
can be re-run safely.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import select

from app.services.iceberg.data_loader import IcebergDataLoader
from app.config import load_datasets, load_vessels

from database.database import SessionLocal
from database.models import Iceberg, IcebergObservation, VesselRecord
from services.data_paths import ICEBERG_CSV

logger = logging.getLogger("dss.services.seeding")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def seed_icebergs() -> int:
    """Upsert latest iceberg positions plus one observation each."""
    items = IcebergDataLoader(data_path=str(ICEBERG_CSV)).load()
    count = 0
    with SessionLocal() as db:
        for raw in items:
            iceberg_id = raw.get("iceberg_id")
            if not iceberg_id:
                continue
            last_observed = raw.get("last_observed")
            if isinstance(last_observed, str):
                try:
                    last_observed = datetime.fromisoformat(last_observed)
                except Exception:
                    last_observed = None
            classification = str(raw.get("classification", "unknown"))
            demo = classification == "synthetic_demo"
            existing = db.get(Iceberg, iceberg_id)
            if existing is None:
                db.add(
                    Iceberg(
                        iceberg_id=iceberg_id,
                        latitude=raw.get("latitude", 0.0),
                        longitude=raw.get("longitude", 0.0),
                        length_km=raw.get("length_km"),
                        width_km=raw.get("width_km"),
                        last_observed=last_observed,
                        source=str(raw.get("source", "")),
                        classification=classification,
                        demo=demo,
                    )
                )
            else:
                existing.latitude = raw.get("latitude", existing.latitude)
                existing.longitude = raw.get("longitude", existing.longitude)
                existing.last_observed = last_observed or existing.last_observed
                existing.classification = classification
                existing.demo = demo
                existing.updated_at = _utcnow()

            has_obs = db.scalar(
                select(IcebergObservation.id)
                .where(IcebergObservation.iceberg_id == iceberg_id)
                .limit(1)
            )
            if has_obs is None:
                db.add(
                    IcebergObservation(
                        iceberg_id=iceberg_id,
                        timestamp=last_observed or _utcnow(),
                        latitude=raw.get("latitude", 0.0),
                        longitude=raw.get("longitude", 0.0),
                        source=str(raw.get("source", "")),
                        demo=demo,
                    )
                )
            count += 1
        db.commit()
    return count


def seed_vessels() -> int:
    """Upsert fleet catalogue into the vessels table."""
    vessels = load_vessels()
    count = 0
    with SessionLocal() as db:
        for v in vessels:
            vessel_id = str(v.get("vessel_id") or v.get("id") or "")
            if not vessel_id:
                continue
            existing = db.get(VesselRecord, vessel_id)
            data = {
                "vessel_id": vessel_id,
                "name": str(v.get("name", "")),
                "vessel_type": str(v.get("type", "")),
                "ice_class": v.get("ice_class"),
                "cruise_speed_knots": v.get("cruise_speed_knots"),
                "max_speed_knots": v.get("max_speed_knots"),
                "fuel_rate_lph": v.get("fuel_rate_lph"),
                "fuel_capacity_tons": v.get("fuel_capacity_tons"),
                "config_json": v,
            }
            if existing is None:
                db.add(VesselRecord(**data))
            else:
                for k, val in data.items():
                    setattr(existing, k, val)
                existing.updated_at = _utcnow()
            count += 1
        db.commit()
    return count


def seed_all() -> None:
    """Run all bootstrap seeds (idempotent)."""
    from services.dataset_service import sync_database
    from services.analytics_service import seed_model_metrics_from_artifacts

    sync_database()
    seed_model_metrics_from_artifacts()
    n_ice = seed_icebergs()
    n_vessels = seed_vessels()
    logger.info(
        "Seeded %d icebergs (observations written) and %d vessels; dataset/DB metadata synced.",
        n_ice,
        n_vessels,
    )