"""Application configuration and settings."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = DATA_DIR / "config"
SAMPLE_DIR = DATA_DIR / "sample"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    APP_NAME: str = "Antarctic Navigation DSS"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DATABASE_URL: str = "sqlite:///./antarctic_dss.db"
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"
    API_V1_PREFIX: str = "/api/v1"
    LOG_LEVEL: str = "INFO"

    DATA_MODE: str = "real"
    DEMO_MODE: str = "off"
    LAND_MASK_FILE: str | None = None
    SEA_ICE_MODEL: str = "persistence"
    ICEBERG_MODEL: str = "persistence"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()


def cors_origins_list() -> list[str]:
    """Return configured CORS origins from the shared comma-separated setting."""
    return [origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()]


def load_config(filename: str) -> dict[str, Any]:
    """Load a JSON configuration file from the config directory."""
    filepath = CONFIG_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    with open(filepath, "r") as f:
        return json.load(f)


def load_ports() -> list[dict[str, Any]]:
    """Load port configuration."""
    data = load_config("ports.json")
    return data.get("ports", [])


def load_research_centers() -> list[dict[str, Any]]:
    """Load research center configuration."""
    data = load_config("research_centers.json")
    return data.get("research_centers", [])


def load_vessels() -> list[dict[str, Any]]:
    """Load vessel configuration."""
    data = load_config("vessels.json")
    return data.get("vessels", [])


def load_datasets() -> dict[str, Any]:
    """Load dataset configuration."""
    data = load_config("datasets.json")
    return data.get("datasets", {})


def load_simulation_config() -> dict[str, Any]:
    """Load simulation configuration."""
    data = load_config("simulation.json")
    return data.get("simulation", {})


def load_navigation_config() -> dict[str, Any]:
    """Load navigation decision-support engine configuration."""
    data = load_config("navigation.json")
    return data.get("navigation", {})
