"""SQLAlchemy database setup for the DSS REST API (SQLite initially).

The engine and session factory are created once at import time. ``URL`` is
taken from ``config.settings.DATABASE_URL`` so tests and deployments can point
at a throwaway SQLite file (or later a PostgreSQL/PostGIS instance).
"""
from __future__ import annotations

from collections.abc import Iterator, Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def _connect_args(url: str) -> dict:
    """SQLite-specific engine arguments."""
    if url.startswith("sqlite"):
        return {"check_same_thread": False}
    return {}


engine = create_engine(
    settings.DATABASE_URL,
    connect_args=_connect_args(settings.DATABASE_URL),
    echo=False,
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables. Import ``database.models`` first so the ORM models
    are registered on ``Base.metadata``."""
    from database import models  # noqa: F401

    Base.metadata.create_all(bind=engine)