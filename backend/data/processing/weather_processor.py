"""ERA5 surface weather preprocessing.

Reads pipeline-processed or raw NetCDF with u10, v10, t2m, msl;
standardises units to the canonical MKS scheme (m s-1, K, Pa); aligns
time; fills isolated NaN gaps; optionally resamples to daily means; and
writes the clean output plus a processing report.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from data_pipeline.common.logging_setup import get_logger
from data_pipeline.common.netcdf_utils import write_netcdf
from data_pipeline.common.schemas import WEATHER_SPEC

from data.processing._common import (
    ProcessingReport,
    antarctic_clip,
    check_units,
    coverage_report,
    daily_resample,
    fill_gaps_along_time,
    standardize_coords,
    write_report,
)

log = get_logger(__name__)

VARIABLES = ("u10", "v10", "t2m", "msl")
# unit corrections applied when a source file uses an alternative scale
UNIT_FIXES = {
    # variable -> (source unit, factor to canonical unit, canonical unit)
    "u10": {"ms": 1.0, "m/s": 1.0, "m s-1": 1.0, "knots": 0.514444, "km/h": 0.277778},
    "v10": {"ms": 1.0, "m/s": 1.0, "m s-1": 1.0, "knots": 0.514444, "km/h": 0.277778},
    "t2m": {
        "K": 1.0,
        "degC": (1.0, 273.15),
        "C": (1.0, 273.15),
        "degree_Celsius": (1.0, 273.15),
    },
    "msl": {"Pa": 1.0, "Pa": 1.0, "hPa": 100.0, "mbar": 100.0, "mb": 100.0},
}


def _normalize_units(ds: xr.Dataset, variable: str) -> tuple[xr.Dataset, str | None]:
    """Convert a weather variable to canonical units; returns adjustment note."""
    spec = WEATHER_SPEC.variable(variable)
    if spec is None or variable not in ds:
        return ds, None
    actual = ds[variable].attrs.get("units", None)
    canonical = spec.units
    if actual == canonical:
        return ds, None
    fixes = UNIT_FIXES.get(variable, {})
    target = fixes.get(actual)
    if target is None:
        return ds, f"{variable}: unknown units `{actual}`, left unchanged"
    if isinstance(target, tuple):
        scale, offset = target
        ds[variable] = ds[variable].astype("float32") * scale + offset
    else:
        ds[variable] = ds[variable].astype("float32") * target
    ds[variable].attrs["units"] = canonical
    return ds, f"{variable}: converted {actual} -> {canonical}"


class WeatherProcessor:
    """Clean ERA5 surface weather for ML training."""

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        reports_dir: Path,
        *,
        resample_daily: bool = True,
        lat_min: float = -90.0,
        lat_max: float = -50.0,
    ) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.reports_dir = Path(reports_dir)
        self.resample_daily = resample_daily
        self.lat_min = lat_min
        self.lat_max = lat_max

    def process(self) -> Path:
        log.info("Weather processing: %s", self.input_path)
        ds = xr.open_dataset(self.input_path)
        ds = standardize_coords(ds)

        report = ProcessingReport(dataset="weather")
        warnings: list[str] = []
        missing_info: dict[str, Any] = {}
        unit_checks: list[dict[str, Any]] = []

        # ── 1. Unit standardisation ───────────────────────────────
        for var in VARIABLES:
            spec = WEATHER_SPEC.variable(var)
            if spec is None or var not in ds:
                if var not in ds:
                    warnings.append(f"Variable `{var}` not found in dataset")
                continue
            note = None
            ds, note = _normalize_units(ds, var)
            uc = check_units(ds, var, spec.units)
            unit_checks.append(
                {
                    "variable": var,
                    "expected": spec.units,
                    "actual": ds[var].attrs.get("units", None),
                    "conforms": uc is None or uc.get("conforms", False),
                    "adjustment": note,
                }
            )
            if note and "unchanged" not in note:
                warnings.append(note)

        # ── 2. Missing data ───────────────────────────────────────
        for var in VARIABLES:
            if var not in ds:
                continue
            vals = ds[var].values
            n_total = int(vals.size)
            n_nan = int(np.isnan(vals).sum())
            missing_info[var] = {
                "count": n_nan,
                "total": n_total,
                "pct": round(n_nan / n_total * 100, 2) if n_total else 0.0,
            }
        present = [v for v in VARIABLES if v in ds]
        if any(missing_info.get(v, {}).get("count", 0) > 0 for v in present):
            ds = fill_gaps_along_time(ds, present, max_gap=2)
            for var in present:
                old_n = missing_info[var]["count"]
                new_n = int(np.isnan(ds[var].values).sum())
                if new_n < old_n:
                    warnings.append(
                        f"Filled {old_n - new_n} NaNs in `{var}` via linear time interpolation"
                    )
                    missing_info[var]["count"] = new_n
                    total = missing_info[var]["total"]
                    missing_info[var]["pct"] = round(new_n / total * 100, 2) if total else 0.0

        # ── 3. Antarctic clip + daily resample ────────────────────
        before_lat = int(ds.sizes.get("latitude", 0))
        ds = antarctic_clip(ds, self.lat_min, self.lat_max)
        after_lat = int(ds.sizes.get("latitude", 0))
        if before_lat - after_lat > 0:
            warnings.append(
                f"Dropped {before_lat - after_lat} latitude rows outside Antarctic band"
            )

        if self.resample_daily and "time" in ds.dims:
            before_t = len(ds.time)
            ds_daily = daily_resample(ds, present)
            after_t = len(ds_daily.time)
            if after_t < before_t:
                warnings.append(f"Resampled {before_t} -> {after_t} daily time steps")
                ds = ds_daily

        # ── 4. Final attrs ────────────────────────────────────────
        ds.attrs["title"] = "ERA5 Antarctic surface weather (processed for ML)"
        ds.attrs["processing_note"] = (
            "Standardised MKS units, Antarctic-clipped, gaps filled"
        )

        shape = {
            dim: int(ds.sizes.get(dim, 0))
            for dim in ("time", "latitude", "longitude")
            if dim in ds.dims
        }
        report.shape = shape
        report.missing = missing_info
        report.units = unit_checks
        report.coverage = coverage_report(ds)
        report.warnings = warnings
        report.status = "ok" if not warnings else "completed_with_warnings"

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        write_netcdf(ds, self.output_path)
        write_report(report, self.reports_dir, "weather_processing")
        log.info("Weather done: %s | shape %s", self.output_path, shape)
        ds.close()
        return self.output_path