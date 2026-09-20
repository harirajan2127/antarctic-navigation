"""Top-level application configuration for the DSS REST API.

Settings are read from environment variables and an optional ``backend/.env``
file (see the repository root ``.env.example`` for the full reference).

Secrets such as remote-data API credentials are only ever available here,
inside the process. They are never included in API responses.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Classification strings used by the data loaders. Anything that is NOT
# synthetic demo data is treated as potentially real (pipeline_data / raw).
SYNTHETIC_DEMO = "synthetic_demo"
PIPELINE_DATA = "pipeline_data"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAND_MASK_FILE = PROJECT_ROOT / "Real data" / "processed" / "bathymetry" / "land_ocean_mask.json"
DEFAULT_PRODUCTION_FRONTEND_ORIGIN = "https://antarctic-navigation-dss.vercel.app"

DEMO_WARNING = (
    "DEMO MODE: these results are based on synthetic demo data used for "
    "development and testing. They carry NO information about real-world "
    "conditions and must not be used for actual navigation decisions."
)

REAL_DATA_WARNING = (
    "Real data loaded. Predictions are estimates based on processed observations "
    "and should not be the sole basis for navigation safety decisions."
)


class Settings(BaseSettings):
    """Environment-driven application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # --- Application ---
    APP_NAME: str = "Antarctic Navigation Decision Support System"
    APP_VERSION: str = "0.2.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # --- Database ---
    DATABASE_URL: str = "sqlite:///./antarctic_dss.db"

    # --- CORS ---
    # Comma-separated list of allowed frontend origins.
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"
    FRONTEND_URL: str | None = None

    # Real-data-only mode for the operational DSS.
    # Demo and mock fallback behavior is intentionally disabled in production.
    DEMO_MODE: str = "off"

    # --- Data source mode ---
    # Production must remain in real-data mode only. Missing real assets are
    # reported as unavailable instead of substituting synthetic or mock data.
    DATA_MODE: str = "real"

    # --- Land / coastline layer for navigation (NetCDF/GeoTIFF/.npy) ---
    # When unset, the routing grid conservatively treats the whole region as
    # open water and this limitation is surfaced on every optimized route.
    LAND_MASK_FILE: str | None = str(DEFAULT_LAND_MASK_FILE)

    # --- Research-station ocean approach ---
    # Research centers that lie on land keep the land mask fully active; route
    # generation snaps to the nearest navigable ocean cell. When that approach
    # is farther than this threshold (km), the route is still generated but a
    # warning states the real offshore distance explicitly.
    RESEARCH_STATION_APPROACH_DISTANCE_KM: float = 100.0

    # --- Model selection (persistence | random_forest | lstm / convlstm) ---
    SEA_ICE_MODEL: str = "persistence"
    ICEBERG_MODEL: str = "random_forest"

    # --- Evaluation targets (displayed separately from measured scores) ---
    TWO_HOUR_EVALUATION_TARGET_PERCENT: float = 10.0
    LAND_MASK_EVALUATION_TARGET_PERCENT: float = 5.0

    # --- Optional local data overrides ---
    SEA_ICE_NETCDF_PATH: str | None = None
    ICEBERG_DATA_PATH: str | None = None

    # --- Defaults ---
    DEFAULT_VESSEL_ID: str = "polar_explorer"
    MODEL_DIR: str = "models"

    # --- Remote data credentials (kept in the process only, never sent out) ---
    COPERNICUS_USERNAME: str | None = None
    COPERNICUS_PASSWORD: str | None = None
    CDS_API_KEY: str | None = None

    # --- AI assistant / LLM provider ---
    # "auto"  -> pick openai / anthropic / ollama / none from available keys
    # "openai"/"anthropic"/"ollama" -> force a provider
    # "none"  -> always use the built-in template answers (no external call)
    ASSISTANT_PROVIDER: str = "auto"
    OPENAI_API_KEY: str | None = None
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-4o-mini"
    ANTHROPIC_API_KEY: str | None = None
    ANTHROPIC_MODEL: str = "claude-3-5-haiku-latest"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:3b"
    # Optional model override that wins over per-provider defaults.
    ASSISTANT_MODEL: str | None = None

    @property
    def cors_origins_list(self) -> list[str]:
        """Parsed list of allowed CORS origins."""
        origins = [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]
        if self.FRONTEND_URL and self.FRONTEND_URL.strip() not in origins:
            origins.append(self.FRONTEND_URL.strip().rstrip("/"))
        if DEFAULT_PRODUCTION_FRONTEND_ORIGIN not in origins:
            origins.append(DEFAULT_PRODUCTION_FRONTEND_ORIGIN)
        return origins

    @property
    def demo_forced(self) -> bool:
        """Production mode never enables synthetic demo behaviour."""
        return False

    @property
    def data_mode_real(self) -> bool:
        """True when real data is required (never fall back to synthetic)."""
        return self.DATA_MODE.strip().lower() == "real"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings singleton."""
    return Settings()


settings = get_settings()


def effective_demo(classification: str | None) -> bool:
    """Production mode intentionally never reports synthetic demo data as live."""
    return False