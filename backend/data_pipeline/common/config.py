"""Pipeline settings and credential handling.

All credentials and endpoints are read from environment variables (via a
``.env`` file or the shell). No credentials are hardcoded in this project.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings

BACKEND_DIR = Path(__file__).resolve().parents[2]
DATASETS_ROOT = BACKEND_DIR / "datasets"


class PipelineSettings(BaseSettings):
    """Environment-driven settings for the data pipeline."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # --- Storage locations ---
    DATA_RAW_ROOT: Path = DATASETS_ROOT / "raw"
    DATA_PROCESSED_ROOT: Path = DATASETS_ROOT / "processed"
    DATA_REPORT_ROOT: Path = DATASETS_ROOT / "reports"
    DATA_IMPORT_ROOT: Path = DATASETS_ROOT / "imports"

    # --- Download behaviour ---
    PIPELINE_RETRIES: int = 3
    PIPELINE_RETRY_BACKOFF_SECONDS: float = 5.0
    PIPELINE_TIMEOUT_SECONDS: float = 120.0
    PIPELINE_SKIP_EXISTING: bool = True
    PIPELINE_CHUNK_BYTES: int = 1 << 20
    PIPELINE_MIN_VALID_BYTES: int = 1024

    # --- Antarctic bounding box (used for clip/validation) ---
    ANTARCTIC_LAT_MIN: float = -90.0
    ANTARCTIC_LAT_MAX: float = -50.0
    ANTARCTIC_LON_MIN: float = -180.0
    ANTARCTIC_LON_MAX: float = 180.0

    # --- NSIDC (Sea Ice Index G02202 / G10016) ---
    NSIDC_API_TOKEN: str = ""
    NSIDC_SEA_ICE_URL_PATTERN: str = ""
    NSIDC_SEA_ICE_BASE_URL: str = (
        "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0051_gsfc_nasateam_seaice/final/gsfc/south/"
    )

    # --- Copernicus Marine (ocean + optionally sea ice) ---
    COPERNICUS_MOTU_URL: str = "https://motu.mercator-ocean.fr/motu-web/Motu"
    COPERNICUS_MOTU_USER: str = ""
    COPERNICUS_MOTU_PASS: str = ""
    COPERNICUS_OCEAN_DATASET_ID: str = "GLOBAL_ANALYSISFORECAST_PHY_001_024-TDS"
    COPERNICUS_SEAICE_DATASET_ID: str = "SEAICE_GLO_SEAICE_L4_NRT_OBSERVATIONS_011_001-TDS"

    # --- Copernicus Climate Data Store (ERA5) ---
    CDS_API_URL: str = "https://cds.climate.copernicus.eu/api"
    CDS_API_KEY: str = ""

    # --- Vessel / journey consumers ---
    DEFAULT_VESSEL_ID: str = "polar_explorer"

    @property
    def has_nsidc_credentials(self) -> bool:
        return bool(self.NSIDC_API_TOKEN)

    @property
    def has_copernicus_credentials(self) -> bool:
        return bool(self.COPERNICUS_MOTU_USER and self.COPERNICUS_MOTU_PASS)

    @property
    def has_cds_credentials(self) -> bool:
        return bool(self.CDS_API_KEY)

    def ensure_dirs(self) -> None:
        for path in (
            self.DATA_RAW_ROOT,
            self.DATA_PROCESSED_ROOT,
            self.DATA_REPORT_ROOT,
            self.DATA_IMPORT_ROOT,
        ):
            path.mkdir(parents=True, exist_ok=True)


# Keep a single cached instance; environment changes require a restart.
_settings: PipelineSettings | None = None


def get_pipeline_settings() -> PipelineSettings:
    global _settings
    if _settings is None:
        _settings = PipelineSettings()
        _settings.ensure_dirs()
    return _settings


def creds_summary() -> dict[str, bool]:
    """Report which credential sets are configured (never their values)."""
    s = get_pipeline_settings()
    return {
        "nsidc_token": s.has_nsidc_credentials,
        "copernicus_marine": s.has_copernicus_credentials,
        "cds_era5": s.has_cds_credentials,
    }