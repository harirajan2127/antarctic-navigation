"""SQLAlchemy ORM models for the DSS database.

Tables:
- icebergs / iceberg_observations : tracked icebergs and their history
- sea_ice_forecasts              : cached sea-ice forecast summaries
- routes                         : persisted route-optimization results
- vessels                        : vessel configuration (fleet catalogue)
- model_metrics                  : trained-model validation metrics
- dataset_metadata               : dataset status metadata/classification
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.database import Base


def utcnow() -> datetime:
    """UTC-aware datetime helper (SQLite stores naive; we normalise here)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def parse_rows_with_date(rows: list[dict[str, Any]], *keys: str) -> list[dict[str, Any]]:
    """Copy of a small helper kept local to avoid cross-package imports."""
    return rows


class Iceberg(Base):
    """Latest-known state of a tracked iceberg."""

    __tablename__ = "icebergs"

    iceberg_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, default="", index=False)
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    length_km: Mapped[float | None] = mapped_column(Float, nullable=True)
    width_km: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_observed: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    source: Mapped[str] = mapped_column(String, default="")
    classification: Mapped[str] = mapped_column(String, default="synthetic_demo")
    demo: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )

    observations: Mapped[list["IcebergObservation"]] = relationship(
        back_populates="iceberg", cascade="all, delete-orphan"
    )


class IcebergObservation(Base):
    """A single historical position fix for an iceberg."""

    __tablename__ = "iceberg_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    iceberg_id: Mapped[str] = mapped_column(
        String, ForeignKey("icebergs.iceberg_id"), index=True
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String, default="")
    demo: Mapped[bool] = mapped_column(Boolean, default=True)

    iceberg: Mapped["Iceberg"] = relationship(back_populates="observations")


class SeaIceForecastRecord(Base):
    """Summary of a computed sea-ice concentration forecast."""

    __tablename__ = "sea_ice_forecasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    forecast_time: Mapped[datetime] = mapped_column(DateTime, index=True)
    valid_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    horizon_hours: Mapped[int] = mapped_column(Integer)
    model: Mapped[str] = mapped_column(String)
    mean_concentration: Mapped[float] = mapped_column(Float)
    max_concentration: Mapped[float] = mapped_column(Float)
    coverage_pct: Mapped[float] = mapped_column(Float)
    demo: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class RouteRecord(Base):
    """A persisted route-optimization result."""

    __tablename__ = "routes"

    route_id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    start_latitude: Mapped[float] = mapped_column(Float)
    start_longitude: Mapped[float] = mapped_column(Float)
    destination_latitude: Mapped[float] = mapped_column(Float)
    destination_longitude: Mapped[float] = mapped_column(Float)
    vessel_id: Mapped[str] = mapped_column(String, index=True)
    preference: Mapped[str] = mapped_column(String, default="recommended")
    distance_km: Mapped[float] = mapped_column(Float)
    distance_nm: Mapped[float] = mapped_column(Float)
    travel_time_hours: Mapped[float] = mapped_column(Float)
    fuel_tons: Mapped[float | None] = mapped_column(Float, nullable=True)
    risk_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    risk_level: Mapped[str | None] = mapped_column(String, nullable=True)
    warnings: Mapped[list[Any]] = mapped_column(JSON, default=list)
    coordinates: Mapped[list[Any]] = mapped_column(JSON, default=list)
    waypoints: Mapped[list[Any]] = mapped_column(JSON, default=list)
    alternatives: Mapped[list[Any]] = mapped_column(JSON, default=list)
    demo: Mapped[bool] = mapped_column(Boolean, default=True)


class VesselRecord(Base):
    """Vessel configuration entry from the fleet catalogue."""

    __tablename__ = "vessels"

    vessel_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, default="")
    vessel_type: Mapped[str] = mapped_column(String, default="")
    ice_class: Mapped[str | None] = mapped_column(String, nullable=True)
    cruise_speed_knots: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_speed_knots: Mapped[float | None] = mapped_column(Float, nullable=True)
    fuel_rate_lph: Mapped[float | None] = mapped_column(Float, nullable=True)
    fuel_capacity_tons: Mapped[float | None] = mapped_column(Float, nullable=True)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )


class ModelMetric(Base):
    """A single validation metric for a trained model."""

    __tablename__ = "model_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pipeline: Mapped[str] = mapped_column(String, index=True)
    model: Mapped[str] = mapped_column(String, index=True)
    metric_name: Mapped[str] = mapped_column(String)
    metric_value: Mapped[float] = mapped_column(Float)
    split: Mapped[str | None] = mapped_column(String, nullable=True)
    demo: Mapped[bool] = mapped_column(Boolean, default=True)
    trained_on: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    source_file: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class DatasetRecord(Base):
    """Status/classification metadata for a known dataset."""

    __tablename__ = "dataset_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_name: Mapped[str] = mapped_column(String, index=True)
    source_id: Mapped[str | None] = mapped_column(String, nullable=True)
    source_name: Mapped[str | None] = mapped_column(String, nullable=True)
    classification: Mapped[str] = mapped_column(String, default="unclassified")
    status: Mapped[str] = mapped_column(String, default="missing")
    file_path: Mapped[str | None] = mapped_column(String, nullable=True)
    records: Mapped[int | None] = mapped_column(Integer, nullable=True)
    demo: Mapped[bool] = mapped_column(Boolean, default=True)
    notes: Mapped[str | None] = mapped_column(String, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )