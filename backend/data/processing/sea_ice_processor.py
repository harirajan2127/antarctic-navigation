"""Sea-ice concentration preprocessing.

Reads pipeline-processed or raw NetCDF, standardises coordinates, clips
to the Antarctic region, validates the [0, 1] concentration range,
optionally resamples to daily means, fills isolated NaN gaps with
linear interpolation along time, and writes the clean output plus a
full processing report.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from data_pipeline.common.logging_setup import get_logger
from data_pipeline.common.netcdf_utils import write_netcdf
from data_pipeline.common.schemas import SEA_ICE_SPEC

from data.processing._common import (
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    ProcessingReport,
    antarctic_clip,
    check_units,
    coverage_report,
    daily_resample,
    fill_gaps_along_time,
    standardize_coords,
    value_range_stats,
    write_report,
)

log = get_logger(__name__)

VAR = "sea_ice_concentration"


class SeaIceProcessor:
    """Clean sea-ice concentration for ML training."""

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        reports_dir: Path,
        *,
        lat_min: float = LAT_MIN,
        lat_max: float = LAT_MAX,
        lon_min: float = LON_MIN,
        lon_max: float = LON_MAX,
        resample_daily: bool = True,
    ) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.reports_dir = Path(reports_dir)
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.resample_daily = resample_daily

    def process(self) -> Path:
        """Run the full sea-ice processing pipeline. Returns output path."""
        log.info("Sea-ice processing: %s", self.input_path)
        ds = xr.open_dataset(self.input_path)
        ds = standardize_coords(ds)

        report = ProcessingReport(dataset="sea_ice")
        warnings: list[str] = []
        missing_info: dict[str, Any] = {}
        unit_checks: list[dict[str, Any]] = []

        # ── 1. Unit conformity ────────────────────────────────────
        expected = SEA_ICE_SPEC.variable(VAR).units
        uc = check_units(ds, VAR, expected)
        unit_checks.append(
            {
                "variable": VAR,
                "expected": expected,
                "actual": ds[VAR].attrs.get("units", None),
                "conforms": uc is None or uc.get("conforms", False),
            }
        )
        if uc and not uc.get("conforms", False):
            warnings.append(uc["message"])
            ds[VAR].attrs["units"] = expected

        # ── 2. Missing data ───────────────────────────────────────
        vals = ds[VAR].values
        n_total = int(vals.size)
        n_nan = int(np.isnan(vals).sum())
        missing_info[VAR] = {
            "count": n_nan,
            "total": n_total,
            "pct": round(n_nan / n_total * 100, 2) if n_total else 0.0,
        }
        if n_nan > 0:
            warnings.append(f"{n_nan}/{n_total} missing ({missing_info[VAR]['pct']}%)")
            ds = fill_gaps_along_time(ds, [VAR], max_gap=1)
            remaining = int(np.isnan(ds[VAR].values).sum())
            if remaining < n_nan:
                warnings.append(
                    f"Filled {n_nan - remaining} NaNs via linear time interpolation"
                )

        # ── 3. Range validation ───────────────────────────────────
        vr = value_range_stats(ds[VAR].values, (0.0, 1.0), VAR)
        if vr["out_of_range"] > 0:
            warnings.append(
                f"Clipped {vr['out_of_range']} out-of-range SIC values to [0, 1]"
            )
        ds[VAR] = ds[VAR].clip(0.0, 1.0)

        # ── 4. Antarctic spatial clip ─────────────────────────────
        before_lat = int(ds.sizes.get("latitude", 0))
        ds = antarctic_clip(ds, self.lat_min, self.lat_max)
        after_lat = int(ds.sizes.get("latitude", 0))
        dropped = before_lat - after_lat
        if dropped > 0:
            warnings.append(
                f"Dropped {dropped} latitude rows outside [{self.lat_min}, {self.lat_max}]"
            )

        # ── 5. Daily resample ─────────────────────────────────────
        if self.resample_daily and "time" in ds.dims:
            before_t = len(ds.time)
            ds_daily = daily_resample(ds, [VAR])
            after_t = len(ds_daily.time)
            if after_t < before_t:
                warnings.append(f"Resampled {before_t} -> {after_t} daily time steps")
                ds = ds_daily

        # ── 6. Final attrs ────────────────────────────────────────
        ds.attrs["title"] = "Antarctic sea-ice concentration (processed for ML)"
        ds.attrs["processing_note"] = "Standardised coordinates, range-validated, Antarctic-clipped"
        ds[VAR].attrs["units"] = "fraction"
        ds[VAR].attrs["long_name"] = SEA_ICE_SPEC.variable(VAR).long_name

        # ── 7. Shape + coverage ───────────────────────────────────
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

        # ── 8. Write ──────────────────────────────────────────────
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        write_netcdf(ds, self.output_path)
        write_report(report, self.reports_dir, "sea_ice_processing")
        log.info("Sea-ice done: %s | shape %s", self.output_path, shape)
        ds.close()
        return self.output_path
