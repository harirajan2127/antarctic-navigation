"""Health check endpoint."""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter

from config import settings
from database.database import SessionLocal
from services.data_paths import OCEAN_NETCDF, SEA_ICE_NETCDF, WEATHER_NETCDF
from services.iceberg_service import iceberg_service
from ml.iceberg.runtime import get_runtime
from schemas.models import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])
system_router = APIRouter(prefix="/system", tags=["system"])


def _db_check() -> str:
    try:
        with SessionLocal() as db:
            from sqlalchemy import text
            db.execute(text("SELECT 1"))
        return "connected"
    except Exception:
        return "unavailable"


def _file_available(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


@lru_cache(maxsize=1)
def _runtime_status() -> dict[str, object]:
    """Inspect production resources once; never claim unavailable resources are ready."""
    iceberg_count = len(iceberg_service.list_icebergs().get("icebergs", []))
    model_available = False
    configured_model = (settings.ICEBERG_MODEL or "persistence").strip().lower()
    if configured_model in {"random_forest", "lstm"}:
        runtime = get_runtime(configured_model)
        checkpoint = "rf.joblib" if configured_model == "random_forest" else "lstm.pt"
        model_available = (runtime.out_dir / checkpoint).is_file()
    land_mask = settings.LAND_MASK_FILE or ""
    land_mask_path = Path(land_mask)
    return {
        "backend": True,
        "demo_mode": False,
        "real_data_mode": True,
        "sea_ice_data": _file_available(Path(SEA_ICE_NETCDF)),
        "ocean_data": _file_available(Path(OCEAN_NETCDF)),
        "weather_data": _file_available(Path(WEATHER_NETCDF)),
        "iceberg_data": iceberg_count > 0,
        "iceberg_records": iceberg_count,
        "land_mask": _file_available(land_mask_path),
        "iceberg_model": model_available,
        "route_engine": True,
    }


@router.get("", response_model=HealthResponse)
async def health() -> HealthResponse:
    """GET /api/health"""
    resources = _runtime_status()
    return HealthResponse(
        status="ok",
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        database=_db_check(),
        timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
        **{key: value for key, value in resources.items() if key in HealthResponse.model_fields},
    )


@system_router.get("/status")
async def system_status() -> dict[str, object]:
    """GET /api/system/status, without exposing local filesystem paths."""
    return {
        "status": "ok" if all(
            _runtime_status().get(key, False)
            for key in ("sea_ice_data", "ocean_data", "weather_data", "iceberg_data", "land_mask", "route_engine")
        ) else "degraded",
        "version": settings.APP_VERSION,
        **_runtime_status(),
    }