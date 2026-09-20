"""Ocean surface field preprocessing.

Reads pipeline-processed NetCDF containing uo, vo, thetao, so;
standardises coordinates; selects the surface depth level when a depth
dimension is present; validates units against the canonical specs;
optionally aligns time/coordinates to a reference grid; fills isolated
NaN gaps; and writes the clean output plus a processing report.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from data_pipeline.common.logging_setup import get_logger
from data_pipeline.common.netcdf_utils import write_netcdf
from data_pipeline.common.schemas import OCEAN_SPEC

from data.processing._common import (
    LAT_MAX,
    LAT_MIN,
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

VARIABLES = ("uo", "vo", "thetao", "so")


class OceanProcessor:
    """Clean ocean surface fields for ML training."""

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        reports_dir: Path,
        *,
        depth_level: int = 0,
        ref_grid_path: Path | None = None,
        lat_min: float = LAT_MIN,
        lat_max: float = LAT_MAX,
        resample_daily: bool = True,
    ) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.reports_dir = Path(reports_dir)
        self.depth_level = depth_level
        self.ref_grid_path = ref_grid_path
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.resample_daily = resample_daily

    def process(self) -> Path:
        log.info("Ocean processing: %s", self.input_path)
        ds = xr.open_dataset(self.input_path)
        ds = standardize_coords(ds)

        report = ProcessingReport(dataset="ocean")
        warnings: list[str] = []
        missing_info: dict[str, Any] = {}
        unit_checks: list[dict[str, Any]] = []

        # ── 1. Depth selection ────────────────────────────────────
        if "depth" in ds.dims:
            depth_vals = ds["depth"].values
            sel_idx = min(self.depth_level, len(depth_vals) - 1)
            depth_val = float(depth_vals[sel_idx])
            warnings.append(
                f"Selected depth level {sel_idx} ({depth_val}) of {len(depth_vals)}"
            )
            ds = ds.isel(depth=sel_idx)
            if "depth" in ds.coords:
                ds = ds.drop_vars("depth", errors="ignore")

        # ── 2. Unit validation ────────────────────────────────────
        for var in VARIABLES:
            spec = OCEAN_SPEC.variable(var)
            if spec is None or var not in ds:
                if var not in ds:
                    warnings.append(f"Variable `{var}` not found in dataset")
                continue
            uc = check_units(ds, var, spec.units)
            unit_checks.append(
                {
                    "variable": var,
                    "expected": spec.units,
                    "actual": ds[var].attrs.get("units", None),
                    "conforms": uc is None or uc.get("conforms", False),
                }
            )
            if uc and not uc.get("conforms", False):
                warnings.append(uc["message"])
                ds[var].attrs["units"] = spec.units

        # ── 3. Missing data ───────────────────────────────────────
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
        nvars_present = [v for v in VARIABLES if v in ds]
        if any(missing_info.get(v, {}).get("count", 0) > 0 for v in nvars_present):
            ds = fill_gaps_along_time(ds, nvars_present, max_gap=2)
            filled_any = False
            for var in nvars_present:
                old_n = missing_info[var]["count"]
                new_n = int(np.isnan(ds[var].values).sum())
                if new_n < old_n:
                    filled_any = True
                    warnings.append(
                        f"Filled {old_n - new_n} NaNs in `{var}` via linear time interpolation"
                    )
                    missing_info[var]["count"] = new_n
                    missing_info[var]["pct"] = round(
                        new_n / missing_info[var]["total"] * 100, 2
                    ) if missing_info[var]["total"] else 0.0

        # ── 4. Antarctic clip ─────────────────────────────────────
        before_lat = int(ds.sizes.get("latitude", 0))
        ds = antarctic_clip(ds, self.lat_min)
        after_lat = int(ds.sizes.get("latitude", 0))
        if before_lat - after_lat > 0:
            warnings.append(
                f"Dropped {before_lat - after_lat} latitude rows outside Antarctic band"
            )

        # ── 5. Reference grid alignment ───────────────────────────
        if self.ref_grid_path and Path(self.ref_grid_path).exists():
            ref = standardize_coords(xr.open_dataset(self.ref_grid_path))
            ref_lat = ref["latitude"].values
            ref_lon = ref["longitude"].values
            ref.close()
            ds = ds.interp(
                latitude=ref_lat, longitude=ref_lon, method="nearest"
            )
            warnings.append(
                f"Aligned to reference grid ({len(ref_lat)} lat × {len(ref_lon)} lon)"
            )

        # ── 6. Daily resample ─────────────────────────────────────
        if self.resample_daily and "time" in ds.dims:
            before_t = len(ds.time)
            ds_daily = daily_resample(ds, [v for v in VARIABLES if v in ds])
            after_t = len(ds_daily.time)
            if after_t < before_t:
                warnings.append(f"Resampled {before_t} -> {after_t} daily time steps")
                ds = ds_daily

        # ── 7. Final attrs ────────────────────────────────────────
        ds.attrs["title"] = "Antarctic ocean surface fields (processed for ML)"
        ds.attrs["processing_note"] = (
            "Standardised coordinates, surface-level selected, Antarctic-clipped"
        )

        # ── 8. Shape + coverage ───────────────────────────────────
        shape = {
            dim: int(ds.sizes.get(dim, 0))
            for dim in ("time", "latitude", "longitude", "depth")
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
        write_report(report, self.reports_dir, "ocean_processing")
        log.info("Ocean done: %s | shape %s", self.output_path, shape)
        ds.close()
        return self.output_path
