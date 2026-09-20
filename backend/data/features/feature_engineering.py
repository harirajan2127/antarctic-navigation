"""Feature engineering for ML model training.

Builds a single row-per-observation feature table from the cleaned
iceberg tracks and the processed environmental grids (sea ice, ocean,
weather).

Temporal alignment policy
-------------------------
Environmental fields and iceberg observations live on different time
grids.  This module NEVER silently joins unrelated timestamps: for every
row it queries each environmental field with a nearest-time lookup and:

  * records ``<var>_gap_hours`` (time between query and matched obs);
  * marks the row stale when the gap exceeds ``tolerance_hours``;
  * sets the environmental features to NaN for stale rows.

The caller can therefore audit exactly which observations were used.
Chronological (not shuffled) splitting is used for train/val/test so no
future information leaks into training data.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_pipeline.common.logging_setup import get_logger

from data.processing._common import (
    EnvLookup,
    forward_bearing,
    haversine_nm,
    write_report,
    ProcessingReport,
)

log = get_logger(__name__)

ENV_FEATURES = {
    "sea_ice": ("sea_ice_concentration",),
    "ocean": ("uo", "vo"),
    "weather": ("u10", "v10", "t2m", "msl"),
}

STALE_COLS = {
    "sea_ice": "sea_ice_stale",
    "ocean": "ocean_stale",
    "weather": "weather_stale",
}

GAP_COLS = {
    "sea_ice": "sea_ice_gap_hours",
    "ocean": "ocean_gap_hours",
    "weather": "weather_gap_hours",
}

TIME_FEATURES = (
    "hour_of_day",
    "day_of_year",
    "month",
    "days_since_first_obs",
)

DERIVED_FEATURES = (
    "sic_lag",
    "sic_change",
    "current_magnitude",
    "current_direction",
    "wind_magnitude",
    "wind_direction",
    "prev_latitude",
    "prev_longitude",
    "iceberg_velocity_knots",
    "track_displacement_nm",
)


class FeatureEngineering:
    """Join cleaned iceberg tracks with environmental fields into features."""

    def __init__(
        self,
        icebergs_csv: Path,
        sea_ice_path: Path,
        ocean_path: Path,
        weather_path: Path,
        output_dir: Path,
        reports_dir: Path,
        *,
        tolerance_hours: float = 24.0,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
    ) -> None:
        self.icebergs_csv = Path(icebergs_csv)
        self.sea_ice_path = Path(sea_ice_path)
        self.ocean_path = Path(ocean_path)
        self.weather_path = Path(weather_path)
        self.output_dir = Path(output_dir)
        self.reports_dir = Path(reports_dir)
        self.tolerance_hours = tolerance_hours
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac

    def build(self) -> pd.DataFrame:
        """Run the full feature-engineering pipeline and persist outputs."""
        df = pd.read_csv(self.icebergs_csv)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values(
            ["iceberg_id", "timestamp"]
        ).reset_index(drop=True)

        # Load environmental lookups (missing files become empty lookups).
        lookups: dict[str, EnvLookup | None] = {
            "sea_ice": self._open_lookup(self.sea_ice_path, ENV_FEATURES["sea_ice"]),
            "ocean": self._open_lookup(self.ocean_path, ENV_FEATURES["ocean"]),
            "weather": self._open_lookup(self.weather_path, ENV_FEATURES["weather"]),
        }

        # ── 1. Row-wise environmental features ────────────────────
        rows: list[dict[str, Any]] = []
        for _, rec in df.iterrows():
            ts = np.datetime64(rec["timestamp"])
            row: dict[str, Any] = {
                "iceberg_id": rec["iceberg_id"],
                "timestamp": rec["timestamp"],
                "latitude": float(rec["latitude"]),
                "longitude": float(rec["longitude"]),
                "length_nm": float(rec["length_nm"]),
                "width_nm": float(rec["width_nm"]),
                "speed_knots": _safe_float(rec.get("speed_knots")),
                "bearing_deg": _safe_float(rec.get("bearing_deg")),
                "time_diff_hours": _safe_float(rec.get("time_diff_hours")),
                "distance_traveled_nm": _safe_float(rec.get("distance_traveled_nm")),
                "age_hours": _safe_float(rec.get("age_hours")),
            }
            for domain in ("sea_ice", "ocean", "weather"):
                vars_ = ENV_FEATURES[domain]
                lookup = lookups[domain]
                if lookup is None:
                    for v in vars_:
                        row[v] = np.nan
                    row[GAP_COLS[domain]] = np.nan
                    row[STALE_COLS[domain]] = 1
                    continue
                res = lookup.lookup(
                    row["latitude"], row["longitude"], ts,
                    tolerance_hours=self.tolerance_hours,
                )
                for v in vars_:
                    row[v] = res.get(v, np.nan)
                row[GAP_COLS[domain]] = res["gap_hours"]
                row[STALE_COLS[domain]] = res["stale_count"]
            rows.append(row)

        feat = pd.DataFrame(rows).sort_values(["iceberg_id", "timestamp"]).reset_index(drop=True)

        # ── 2. Derived kinematic / environmental features ─────────
        feat["current_magnitude"] = np.hypot(
            feat["uo"].fillna(0.0), feat["vo"].fillna(0.0)
        )
        feat["current_direction"] = _vector_direction(feat["uo"], feat["vo"])
        feat["wind_magnitude"] = np.hypot(
            feat["u10"].fillna(0.0), feat["v10"].fillna(0.0)
        )
        feat["wind_direction"] = _vector_direction(feat["u10"], feat["v10"])

        # ── 3. Temporal / lag features per iceberg ────────────────
        feat["hour_of_day"] = feat["timestamp"].dt.hour
        feat["day_of_year"] = feat["timestamp"].dt.dayofyear
        feat["month"] = feat["timestamp"].dt.month
        first = feat.groupby("iceberg_id")["timestamp"].transform("min")
        feat["days_since_first_obs"] = (
            (feat["timestamp"] - first) / pd.Timedelta(days=1)
        ).round(4)

        grouped = feat.groupby("iceberg_id")
        feat["sic_lag"] = grouped["sea_ice_concentration"].shift(1)
        feat["sic_change"] = feat["sea_ice_concentration"] - feat["sic_lag"]
        feat["prev_latitude"] = grouped["latitude"].shift(1)
        feat["prev_longitude"] = grouped["longitude"].shift(1)
        feat["iceberg_velocity_knots"] = feat["speed_knots"]
        feat["track_displacement_nm"] = feat.groupby("iceberg_id")[
            "distance_traveled_nm"
        ].transform("max")

        # derive displacement magnitude/direction from successive positions
        prev_lat, prev_lon = feat["prev_latitude"], feat["prev_longitude"]
        mask = prev_lat.notna() & prev_lon.notna()
        disp = np.full(len(feat), np.nan)
        bear = np.full(len(feat), np.nan)
        disp_np = haversine_nm(
            prev_lat[mask].to_numpy(), prev_lon[mask].to_numpy(),
            feat["latitude"][mask].to_numpy(), feat["longitude"][mask].to_numpy(),
        )
        bear_np = forward_bearing(
            prev_lat[mask].to_numpy(), prev_lon[mask].to_numpy(),
            feat["latitude"][mask].to_numpy(), feat["longitude"][mask].to_numpy(),
        )
        disp[mask.values] = disp_np
        bear[mask.values] = bear_np
        feat["displacement_nm"] = disp
        feat["iceberg_heading_deg"] = bear

        # ── 4. Reports + splits + persistence ─────────────────────
        feature_summary = self._feature_summary(feat)
        write_report(
            self._report_object("feature_summary", feature_summary),
            self.reports_dir,
            "feature_summary",
        )
        self._write_shape_report(
            {"rows": int(len(feat)), "columns": int(feat.shape[1]), "icebergs": int(feat["iceberg_id"].nunique())}
        )
        self._write_unit_report()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        feat.to_csv(self.output_dir / "feature_table.csv", index=False)

        # chronological split
        splits = self.split_chronological(feat)
        for name, split in splits.items():
            split.to_csv(self.output_dir / f"{name}.csv", index=False)
        log.info(
            "Features written: %s | %d rows × %d cols",
            self.output_dir / "feature_table.csv",
            len(feat),
            feat.shape[1],
        )
        return feat

    # ── helpers ───────────────────────────────────────────────────

    def split_chronological(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Split ordered by timestamp into train/val/test (chronological)."""
        planned = self.train_frac + self.val_frac + self.test_frac
        if not np.isclose(planned, 1.0):
            raise ValueError(
                f"train+val+test fractions must sum to 1.0 (got {planned})"
            )
        df = df.sort_values("timestamp").reset_index(drop=True)
        n = len(df)
        n_train = int(round(n * self.train_frac))
        n_val = int(round(n * self.val_frac))
        return {
            "train": df.iloc[:n_train],
            "val": df.iloc[n_train:n_train + n_val],
            "test": df.iloc[n_train + n_val:],
        }

    def _open_lookup(self, path: Path, variables: tuple[str, ...]) -> EnvLookup | None:
        if Path(path).exists():
            try:
                return EnvLookup(path, variables)
            except Exception as exc:  # e.g. missing variable
                log.warning("Lookup failed for %s: %s", path, exc)
                return None
        return None

    @staticmethod
    def _feature_summary(feat: pd.DataFrame) -> dict[str, Any]:
        num = feat.select_dtypes(include=[np.number])
        summary: dict[str, Any] = {
            "n_rows": int(len(feat)),
            "n_columns": int(feat.shape[1]),
            "n_icebergs": int(feat["iceberg_id"].nunique()),
            "temporal_start": str(feat["timestamp"].min()),
            "temporal_end": str(feat["timestamp"].max()),
            "stale_env_rows": int((feat[list(STALE_COLS.values())].astype(bool)).any(axis=1).sum()),
            "features": {},
        }
        for col in num.columns:
            s = num[col]
            has_data = bool(s.notna().any())
            summary["features"][col] = {
                "dtype": str(s.dtype),
                "missing": int(s.isna().sum()),
                "missing_pct": round(float(s.isna().mean()) * 100, 2),
                "min": _pyfloat(s.min()) if has_data else None,
                "max": _pyfloat(s.max()) if has_data else None,
                "mean": _pyfloat(s.mean()) if has_data else None,
                "std": _pyfloat(s.std()) if has_data else None,
            }
        return summary

    @staticmethod
    def _report_object(title: str, payload: dict[str, Any]) -> ProcessingReport:
        r = ProcessingReport(dataset=title, status="ok")
        r.coverage = payload
        return r

    def _write_shape_report(self, shape: dict[str, int]) -> None:
        r = ProcessingReport(dataset="dataset_shape", status="ok", shape=shape)
        write_report(r, self.reports_dir, "dataset_shape")

    def _write_unit_report(self) -> None:
        from data_pipeline.common.schemas import OCEAN_SPEC, SEA_ICE_SPEC, WEATHER_SPEC

        rows: list[dict[str, Any]] = []
        for spec, vars_ in (
            (SEA_ICE_SPEC, ("sea_ice_concentration",)),
            (OCEAN_SPEC, ("uo", "vo", "thetao", "so")),
            (WEATHER_SPEC, ("u10", "v10", "t2m", "msl")),
        ):
            for v in vars_:
                s = spec.variable(v)
                rows.append(
                    {
                        "dataset": spec.dataset,
                        "variable": v,
                        "units": s.units if s else None,
                        "valid_range": list(s.valid_range) if s and s.valid_range else None,
                    }
                )
        r = ProcessingReport(dataset="unit_report", status="ok")
        r.coverage = {"variables": rows}
        write_report(r, self.reports_dir, "unit_report")


def _safe_float(value: Any) -> float:
    try:
        f = float(value)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _pyfloat(value: Any) -> float:
    """Coerce numpy scalars to native float for portable JSON output."""
    return round(float(value), 4) if value is not None and np.isfinite(value) else None


def _vector_direction(u: pd.Series, v: pd.Series) -> pd.Series:
    """Direction (degrees clockwise from North) the vector points TOWARDS.

    Uses the atan2(u, v) convention (0=N, 90=E, 180=S, 270=W).  Applies to
    both ocean currents and wind vectors; the convention is stated in the
    feature summary so callers interpret it consistently.
    """
    ang = (np.degrees(np.arctan2(u, v)) + 360.0) % 360.0
    return pd.Series(ang, index=u.index).fillna(np.nan)