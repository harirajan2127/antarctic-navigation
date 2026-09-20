"""Two-hour rolling update support for environmental data.

The navigation system recalculates a route every two simulated hours and
needs environmental fields at arbitrary query times. Remote observation
datasets (e.g. sea-ice) are typically daily, NOT 2-hourly.

This module never pretends daily data is 2-hourly. Instead it:
1. Reports the true cadence of each dataset.
2. Serves the closest valid observation/forecast time with metadata that
   records the ORIGINAL timestamp and the method used (nearest / linear).
3. Produces a freshness report (age of the newest observation).

Interpolation is applied only where scientifically appropriate
(linear interpolation in time for smooth fields such as current velocity,
nearest-observation for categorical/binary fields such as ice edge).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from data_pipeline.common.config import get_pipeline_settings
from data_pipeline.common.logging_setup import get_logger

log = get_logger(__name__)


def default_frequency_hours(times: np.ndarray) -> float | None:
    """Estimate median time step (hours) of a time coordinate."""
    if times.size < 2:
        return None
    diffs = np.diff(np.asarray(times, dtype="datetime64[h]").astype("int64"))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


def _as_datetime64(value: datetime | np.datetime64) -> tuple[np.datetime64, str]:
    """Normalize a query time to (np.datetime64, iso-string)."""
    if isinstance(value, np.datetime64):
        return value, str(value)
    if isinstance(value, datetime):
        return np.datetime64(value.replace(tzinfo=None)), value.isoformat()
    raise TypeError(f"Unsupported query time type: {type(value)!r}")


class EnvironmentalAccessor:
    """Loads processed NetCDF environmental fields plus freshness metadata."""

    def __init__(self, processed_root: Path | None = None) -> None:
        settings = get_pipeline_settings()
        self.processed_root = Path(processed_root or settings.DATA_PROCESSED_ROOT)
        self._cache: dict[str, xr.Dataset] = {}

    def _load(self, dataset: str) -> xr.Dataset:
        if dataset not in self._cache:
            path = self.processed_root / f"{dataset}.nc"
            if not path.exists():
                raise FileNotFoundError(f"Processed dataset not found: {path}")
            self._cache[dataset] = xr.open_dataset(path)
        return self._cache[dataset]

    def cadence(self, dataset: str) -> float | None:
        """True time cadence in hours (from the dataset itself)."""
        ds = self._load(dataset)
        if "time" in ds.coords:
            return default_frequency_hours(ds["time"].values)
        return None

    def first_and_last_time(self, dataset: str) -> tuple[str | None, str | None]:
        ds = self._load(dataset)
        if "time" in ds.coords:
            t = ds["time"].values
            return str(t.min()), str(t.max())
        return None, None

    def age_of_newest_observation(self, dataset: str, now: datetime | None = None) -> timedelta | None:
        """Freshness: age of the newest available timestamp."""
        _, last = self.first_and_last_time(dataset)
        if last is None:
            return None
        last_dt = datetime.fromisoformat(str(last).replace("Z", "+00:00"))
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
        return (now or datetime.now().astimezone()) - last_dt

    def query(
        self,
        dataset: str,
        variable: str,
        query_time: datetime,
        method: str = "nearest",
    ) -> dict[str, Any]:
        """Return an environmental field at ``query_time`` with provenance.

        ``method`` controls interpolation:
          - ``nearest``: the closest observation in time (safe default).
          - ``linear``: linear interpolation in time (only meaningful for
            smooth fields, e.g. current velocity, temperature).

        The result always records which original timestamps bracket the query
        and which method produced the field, so the caller can report data
        freshness honestly.
        """
        ds = self._load(dataset)
        if "time" not in ds.coords or variable not in ds:
            raise KeyError(f"{variable} not available in {dataset}")

        times = ds["time"].values
        target, query_iso = _as_datetime64(query_time)

        # Nearest / bracketing original timestamps.
        idx_nearest = int(np.argmin(np.abs(times - target)))
        nearest_time = str(times[idx_nearest])

        before_mask = times <= target
        after_mask = times >= target
        before_idx = int(np.argmax(before_mask)) if before_mask.any() else None
        after_idx = int(np.argmax(after_mask)) if after_mask.any() else None

        if before_idx == after_idx or target == times[idx_nearest]:
            # Exact hit: no interpolation.
            field = ds[variable].isel(time=idx_nearest)
            meta = {
                "dataset": dataset,
                "variable": variable,
                "query_time": query_iso,
                "original_timestamp": nearest_time,
                "method": "exact_match",
                "source": ds.attrs.get("source_name", "unknown"),
                "is_demo": ds.attrs.get("classification", "") == "synthetic_demo",
                "bracketing_times": [nearest_time, nearest_time],
            }
            return {"values": field.values, "metadata": meta}

        # Interpolation path.
        if method == "nearest":
            field = ds[variable].isel(time=idx_nearest)
            meta = {
                "dataset": dataset,
                "variable": variable,
                "query_time": query_iso,
                "original_timestamp": nearest_time,
                "method": "nearest_in_time",
                "source": ds.attrs.get("source_name", "unknown"),
                "is_demo": ds.attrs.get("classification", "") == "synthetic_demo",
                "bracketing_times": list({str(t) for t in times[[before_idx, after_idx]] if t is not None}),
            }
            return {"values": field.values, "metadata": meta}

        if method == "linear":
            if before_idx is None or after_idx is None:
                field = ds[variable].isel(time=idx_nearest)
                meta = {
                    "dataset": dataset,
                    "variable": variable,
                    "query_time": query_iso,
                    "original_timestamp": nearest_time,
                    "method": "nearest_in_time (edge-of-domain)",
                    "source": ds.attrs.get("source_name", "unknown"),
                    "is_demo": ds.attrs.get("classification", "") == "synthetic_demo",
                    "bracketing_times": None,
                }
                return {"values": field.values, "metadata": meta}

            t0 = times[before_idx]
            t1 = times[after_idx]
            span_h = (t1 - t0) / np.timedelta64(1, "h")
            t_target = float((target - t0) / np.timedelta64(1, "h"))
            w = t_target / span_h if span_h > 0 else 0.0

            f0 = ds[variable].isel(time=before_idx).values
            f1 = ds[variable].isel(time=after_idx).values
            field = f0 * (1.0 - w) + f1 * w

            meta = {
                "dataset": dataset,
                "variable": variable,
                "query_time": query_iso,
                "original_timestamps": [str(t0), str(t1)],
                "method": "linear_in_time",
                "interpolation_weight": round(w, 4),
                "source": ds.attrs.get("source_name", "unknown"),
                "is_demo": ds.attrs.get("classification", "") == "synthetic_demo",
                "bracketing_times": [str(t0), str(t1)],
            }
            return {"values": field, "metadata": meta}

        raise ValueError(f"Unknown interpolation method: {method}")

    def freshness_report(self) -> list[dict[str, Any]]:
        """Describe cadence and age for all processed datasets."""
        report = []
        for path in sorted(self.processed_root.glob("*.nc")):
            dataset = path.stem
            ds = xr.open_dataset(path)
            freq = default_frequency_hours(ds["time"].values) if "time" in ds.coords else None
            report.append(
                {
                    "dataset": dataset,
                    "file": path.name,
                    "classification": ds.attrs.get("classification", "unknown"),
                    "source_name": ds.attrs.get("source_name", "unknown"),
                    "cadence_hours": freq,
                    "note": (
                        f"Data cadence is {freq} hourly; navigation queries use "
                        "the recorded timestamps and are interpolated only when "
                        "scientifically appropriate. Demo data is not real."
                        if freq and freq > 2
                        else "Supports 2-hour cadence."
                    ),
                }
            )
            ds.close()
        return report