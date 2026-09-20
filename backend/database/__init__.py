"""Database package for the DSS REST API.

Exposes the engine, session factory, ORM models, initialisation, and the
FastAPI ``get_db`` dependency.
"""
from database.database import Base, SessionLocal, engine, get_db, init_db
from database.models import (
    DatasetRecord,
    Iceberg,
    IcebergObservation,
    ModelMetric,
    RouteRecord,
    SeaIceForecastRecord,
    VesselRecord,
)

__all__ = [
    "Base",
    "SessionLocal",
    "engine",
    "get_db",
    "init_db",
    "DatasetRecord",
    "Iceberg",
    "IcebergObservation",
    "ModelMetric",
    "RouteRecord",
    "SeaIceForecastRecord",
    "VesselRecord",
]