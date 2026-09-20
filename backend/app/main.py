"""Main FastAPI application entry point."""
from __future__ import annotations

from datetime import datetime, timezone

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import cors_origins_list, settings
from app.models.database import init_db
from app.routes import (
    config_routes,
    iceberg_routes,
    navigation_routes,
    sea_ice_routes,
    vessel_routes,
)
from app.models.schemas import HealthResponse
from api import navigation_routes as route_optimization_api


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "AI-enabled Antarctic sea-ice, iceberg trajectory, and navigation "
        "decision support system. Problem Statement ID: 26059 (MoES/NCPOR)."
    ),
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(config_routes.router, prefix=settings.API_V1_PREFIX)
app.include_router(vessel_routes.router, prefix=settings.API_V1_PREFIX)
app.include_router(navigation_routes.router, prefix=settings.API_V1_PREFIX)
app.include_router(sea_ice_routes.router, prefix=settings.API_V1_PREFIX)
app.include_router(iceberg_routes.router, prefix=settings.API_V1_PREFIX)
app.include_router(route_optimization_api.router, prefix=settings.API_V1_PREFIX)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        timestamp=datetime.now(timezone.utc),
    )


@app.get("/", tags=["system"])
def root():
    """Root endpoint pointing to the interactive docs."""
    return {
        "service": settings.APP_NAME,
        "docs": "/docs",
        "health": "/health",
        "api_prefix": settings.API_V1_PREFIX,
    }