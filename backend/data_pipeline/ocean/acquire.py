"""Oceanographic dataset acquisition and validation.

Real source: Copernicus Marine (GLOBAL_ANALYSISFORECAST_PHY_001_024) with
variables uo (east current), vo (north current), thetao (temperature),
so (salinity). Falls back to labeled synthetic demo data.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from data_pipeline.common import netcdf_utils
from data_pipeline.common.config import get_pipeline_settings
from data_pipeline.common.downloader import DownloadError, RobustDownloader
from data_pipeline.common.logging_setup import get_logger
from data_pipeline.common.netcdf_utils import write_netcdf
from data_pipeline.common.schemas import OCEAN_SPEC
from data_pipeline.common.validation import (
    ValidationResult,
    numeric_range_check,
    validate_missing_values,
)

log = get_logger(__name__)

OCEAN_VARIABLES = ("uo", "vo", "thetao", "so")


def _copernicus_ocean(start: datetime, end: datetime, out_file: Path) -> Path:
    """Download ocean fields from Copernicus Marine via MOTU."""
    try:
        import motuclient  # noqa: F401
    except ModuleNotFoundError as exc:
        raise DownloadError(
            "motuclient is required for Copernicus Marine downloads. "
            "Install with: pip install motuclient"
        ) from exc

    from motuclient import motu_api

    settings = get_pipeline_settings()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "-u", settings.COPERNICUS_MOTU_USER,
        "-p", settings.COPERNICUS_MOTU_PASS,
        "-m", settings.COPERNICUS_MOTU_URL,
        "-d", settings.COPERNICUS_OCEAN_DATASET_ID,
        "-x", str(settings.ANTARCTIC_LON_MIN),
        "-X", str(settings.ANTARCTIC_LON_MAX),
        "-y", str(settings.ANTARCTIC_LAT_MIN),
        "-Y", str(settings.ANTARCTIC_LAT_MAX),
        "-t", start.strftime("%Y-%m-%d %H:00:00"),
        "-T", end.strftime("%Y-%m-%d %H:00:00"),
        *[flag for var in OCEAN_VARIABLES for flag in ("-v", var)],
        "-o", str(out_file.parent),
        "-f", out_file.name,
    ]
    log.info("Requesting Copernicus Marine ocean product via MOTU...")
    motu_api.execute(args)
    return out_file


def ensure_ocean(
    start: datetime | None = None,
    end: datetime | None = None,
    mode: str = "auto",
) -> Path:
    """Return a processed ocean-surface NetCDF file (real or labeled demo)."""
    settings = get_pipeline_settings()
    end = end or datetime.now(timezone.utc)
    start = start or (end - timedelta(days=30))

    processed = settings.DATA_PROCESSED_ROOT / "ocean_surface.nc"
    if processed.exists() and settings.PIPELINE_SKIP_EXISTING:
        log.info("Ocean processed file already exists: %s", processed)
        return processed

    has_creds = settings.has_copernicus_credentials
    if mode == "demo" or (mode == "auto" and not has_creds):
        log.info("Ocean: no live source configured -> generating labeled demo data.")
        return _generate_demo_ocean(start, end, processed)

    if not has_creds:
        raise DownloadError("No Copernicus Marine credentials configured; use --mode demo.")

    raw = settings.DATA_RAW_ROOT / "ocean_copernicus.nc"
    try:
        _copernicus_ocean(start, end, raw)
        return _normalize_ocean(raw, processed)
    except Exception as exc:
        log.error("Copernicus ocean download failed: %s", exc)
        if mode == "real":
            raise
        return _generate_demo_ocean(start, end, processed)


def _normalize_ocean(raw: Path, processed: Path) -> Path:
    ds = xr.open_dataset(raw)
    ds = ds.rename(
        {name: {"latitude": "latitude", "longitude": "longitude"}.get(name, name) for name in ds.variables}
    )
    ds.attrs.update(
        netcdf_utils.stamp_attrs(
            {
                "classification": "real_observation",
                "source_name": "copernicus-marine",
                "product": get_pipeline_settings().COPERNICUS_OCEAN_DATASET_ID,
            }
        )
    )
    write_netcdf(ds, processed)
    return processed


def _generate_demo_ocean(start: datetime, end: datetime, out_file: Path) -> Path:
    if out_file.exists():
        return out_file

    lats = np.arange(-78.0, -50.0, 1.0)
    lons = np.arange(-179.0, 180.0, 1.0)
    times = np.arange(
        np.datetime64(start.date()),
        np.datetime64(end.date()) + np.timedelta64(1, "D"),
        np.timedelta64(1, "D"),
    )
    glat, glon = np.meshgrid(lats, lons, indexing="ij")

    rng = np.random.RandomState(20240101)
    uo = 0.05 * rng.normal(size=(len(times), len(lats), len(lons))).astype("float32")
    vo = 0.05 * rng.normal(size=(len(times), len(lats), len(lons))).astype("float32")
    thetao = np.clip(
        -1.8 + 2.0 * (glat + 90.0) / 40.0,
        -1.8, 2.0,
    ).astype("float32")[None, :, :] + 0.1 * rng.standard_normal(
        (len(times), len(lats), len(lons))
    ).astype("float32")
    so = np.clip(
        33.5 + 0.01 * rng.standard_normal((len(times), len(lats), len(lons))),
        33.0, 34.5,
    ).astype("float32")

    ds = xr.Dataset(
        {
            "uo": (("time", "latitude", "longitude"), uo),
            "vo": (("time", "latitude", "longitude"), vo),
            "thetao": (("time", "latitude", "longitude"), thetao),
            "so": (("time", "latitude", "longitude"), so),
        },
        coords={
            "time": times.astype("datetime64[ns]"),
            "latitude": (("latitude",), lats),
            "longitude": (("longitude",), lons),
        },
        attrs={
            **netcdf_utils.stamp_attrs(netcdf_utils.DEMO_GLOBAL_ATTRS),
            "title": OCEAN_SPEC.title,
            "cadence_hours": str(OCEAN_SPEC.cadence_hours_default),
        },
    )
    ds["uo"].attrs.update(units="m s-1", long_name="Eastward current (synthetic demo), surface")
    ds["vo"].attrs.update(units="m s-1", long_name="Northward current (synthetic demo), surface")
    ds["thetao"].attrs.update(units="K", long_name="Sea surface temperature (synthetic demo)")
    ds["so"].attrs.update(units="psu", long_name="Sea surface salinity (synthetic demo)")
    return write_netcdf(ds, out_file)


def validate(processed_root: Path) -> ValidationResult:
    """Validate the processed ocean-surface dataset."""
    settings = get_pipeline_settings()
    path = Path(processed_root) / "ocean_surface.nc"
    result = ValidationResult(dataset="ocean", file=path.name, status="missing")
    if not path.exists():
        result.add("error", "FILE_MISSING", "Processed ocean NetCDF not found")
        return result

    ds = xr.open_dataset(path)
    result.meta = {
        "classification": ds.attrs.get("classification", "unknown"),
        "source": ds.attrs.get("source_name", "unknown"),
        "cadence_hours": ds.attrs.get("cadence_hours"),
    }
    result.status = "valid"
    if ds.attrs.get("classification") == "synthetic_demo":
        result.add("warning", "DEMO_DATA", "Synthetic demo data; not real measurements.")

    for coord in ("time", "latitude", "longitude"):
        if coord not in ds.coords:
            result.status = "invalid"
            result.add("error", "COORD_MISSING", f"Missing coordinate {coord}")

    for var, lo, hi in (
        ("uo", -5.0, 5.0),
        ("vo", -5.0, 5.0),
        ("thetao", -3.0, 308.0),
        ("so", 0.0, 42.0),
    ):
        if var not in ds:
            result.status = "invalid"
            result.add("error", "VARIABLE_MISSING", f"Missing variable {var}")
            continue
        issues = [
            numeric_range_check(var, ds[var].values, lo, hi),
            validate_missing_values(var, ds[var].values),
        ]
        for iss in issues:
            if iss and iss.severity == "error":
                result.status = "invalid"
            if iss:
                result.add(iss.severity, iss.code, iss.message, iss.detail)

    ds.close()
    return result