"""Iceberg data, prediction, trajectory and distance service.

Wraps ``app.services.iceberg`` behind a stable demo-aware interface. Module-
level singletons cache the data loader output and feature table so repeated
API requests never re-read the CSVs.

The trajectory endpoint returns both historical observations (from the demo
feature table, when available) and persistence/ML predictions for the
requested horizon. Every prediction is clearly marked ``demo`` when the
underlying data is synthetic.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.services.iceberg.data_loader import IcebergDataLoader
from app.services.iceberg.predictor import IcebergTrajectoryService
from ml.iceberg.runtime import get_runtime
from services.distance_calculator import (
    distance_between_icebergs,
    haversine_distance_km,
)

from config import DEMO_WARNING, effective_demo, settings
from services.data_paths import ICEBERG_CSV

logger = logging.getLogger("dss.services.iceberg")

TRAINED_MODELS = ("random_forest", "lstm")

BACKEND_DIR = Path(__file__).resolve().parents[1]
FEATURE_TABLE = BACKEND_DIR / "data" / "processed" / "features" / "feature_table.csv"
REAL_FEATURE_TABLE = BACKEND_DIR / "data" / "processed" / "features" / "feature_table_real.csv"
NM_TO_KM = 1.852


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _parse_ts(val: str | float | None) -> datetime | None:
    if val is None:
        return None
    try:
        from datetime import timezone as _tz
        ts = pd.Timestamp(val)
        return ts.to_pydatetime().replace(tzinfo=_tz.utc).replace(tzinfo=None)
    except Exception:
        return None


class IcebergService:
    """Caches iceberg data and prediction results at instantiation time."""

    def __init__(self) -> None:
        self._loader = IcebergDataLoader(data_path=str(ICEBERG_CSV))
        self._cache: list[dict] = []
        self._feature_cache: pd.DataFrame | None = None
        self._predictor_cache: dict[str, IcebergTrajectoryService] = {}
        self._demo = False
        self._classification = "unknown"
        self._warning: str | None = None

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _load(self) -> list[dict]:
        if not self._cache:
            self._cache = self._loader.load()
            raw_cls = str(
                self._cache[0].get("classification", "unknown") if self._cache else "unknown"
            )
            self._classification = (
                "real_iceberg" if raw_cls == "pipeline_data" else raw_cls
            )
            self._demo = effective_demo(raw_cls)
            self._warning = self._cache[0].get("demo_notice") if self._cache and self._demo else None
        return self._cache

    def _feature_table(self) -> pd.DataFrame:
        if self._feature_cache is None:
            table_path = REAL_FEATURE_TABLE if REAL_FEATURE_TABLE.exists() else FEATURE_TABLE
            if table_path.exists():
                df = pd.read_csv(table_path)
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                self._feature_cache = df.sort_values(["iceberg_id", "timestamp"]).reset_index(drop=True)
            else:
                self._feature_cache = pd.DataFrame()
        return self._feature_cache

    def _predictor(self, model: str) -> IcebergTrajectoryService:
        if model not in self._predictor_cache:
            self._predictor_cache[model] = IcebergTrajectoryService(model_name=model)
        return self._predictor_cache[model]

    def _resolve_model(self, model: str | None) -> str:
        if not model or model == "auto":
            chosen = (settings.ICEBERG_MODEL or "persistence").strip().lower()
            return chosen if chosen else "persistence"
        return model.strip().lower()

    def _trained_predict(self, iceberg_id: str, horizon_hours: int, model: str) -> dict | None:
        """Try the requested trained model runtime for one iceberg; None on failure."""
        runtime = get_runtime(model)
        try:
            out = runtime.predict_iceberg(iceberg_id, horizon_hours)
        except Exception:  # noqa: BLE001
            logger.exception("Iceberg runtime '%s' failed for %s", model, iceberg_id)
            out = None
        if out is not None:
            out["model_kind"] = model
        return out

    def _id_to_record(self) -> dict[str, dict]:
        return {item["iceberg_id"]: item for item in self._load()}

    def _observation_count(self, iceberg_id: str) -> int:
        ft = self._feature_table()
        if ft.empty:
            return 0
        return int((ft["iceberg_id"] == iceberg_id).sum())

    def _observations_for(self, iceberg_id: str) -> list[dict]:
        """Return historical observations from the feature table, as dicts."""
        ft = self._feature_table()
        if ft.empty:
            return []
        subset = ft[ft["iceberg_id"] == iceberg_id][["timestamp", "latitude", "longitude"]].copy()
        if subset.empty:
            return []
        subset = subset.dropna(subset=["timestamp"])
        # Sample 6-hourly positions while retaining the true start and end.
        if len(subset) > 6:
            sampled = subset.iloc[::6, :]
            subset = pd.concat([sampled, subset.iloc[[-1], :]]).drop_duplicates(
                subset=["timestamp"], keep="last"
            )
        return [
            {
                "timestamp": _parse_ts(row["timestamp"]),
                "step": "observation",
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
            }
            for _, row in subset.sort_values("timestamp").iterrows()
        ]

    def _base(self) -> dict:
        return {
            "classification": self._classification,
            "demo": self._demo,
            "warning": self._warning,
        }

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------
    def list_icebergs(self) -> dict:
        items = self._load()
        return {
            "icebergs": [
                {
                    "iceberg_id": item["iceberg_id"],
                    "latitude": item["latitude"],
                    "longitude": item["longitude"],
                    "length_km": item.get("length_km"),
                    "width_km": item.get("width_km"),
                    "last_observed": item.get("last_observed"),
                    "source": item.get("source"),
                    "demo": self._demo,
                }
                for item in items
            ],
            "count": len(items),
            **self._base(),
        }

    def detail(self, iceberg_id: str) -> dict | None:
        items = self._id_to_record()
        item = items.get(iceberg_id)
        if item is None:
            return None
        item.update(self._base())
        item.update({"observation_count": self._observation_count(iceberg_id)})
        return item

    def predict(
        self,
        iceberg_ids: list[str] | None = None,
        horizon_hours: int = 24,
        model: str | None = None,
    ) -> dict:
        model = self._resolve_model(model)
        items = self._load()
        if iceberg_ids:
            items = [item for item in items if item["iceberg_id"] in iceberg_ids]

        predictions = []
        used_trained = False
        if model in TRAINED_MODELS:
            for item in items:
                out = self._trained_predict(item["iceberg_id"], horizon_hours, model)
                if out is not None:
                    used_trained = True
                    predictions.append(
                        {
                            "iceberg_id": out["iceberg_id"],
                            "current_lat": out["current"]["lat"],
                            "current_lon": out["current"]["lon"],
                            "predicted_lat": round(out["final"]["lat"], 4),
                            "predicted_lon": round(out["final"]["lon"], 4),
                            "prediction_hours": horizon_hours,
                            "confidence": out.get("confidence_note"),
                            "demo": self._demo or out.get("demo_only", True),
                        }
                    )
        if len(predictions) < len(items):
            # Fall back to the baseline for any iceberg the trained model
            # could not be applied to.
            remaining = {p["iceberg_id"] for p in predictions}
            fallback_items = [i for i in items if i["iceberg_id"] not in remaining]
            predictor = self._predictor(model if model in TRAINED_MODELS else "persistence")
            fallback = predictor.predict(fallback_items, horizon_hours=horizon_hours)
            for p in fallback:
                p["confidence"] = predictor._predictor.confidence_note()
            predictions.extend(fallback)

        return {
            "icebergs": predictions,
            "model": model,
            "horizon_hours": horizon_hours,
            "trained_used": used_trained,
            **self._base(),
        }

    def trajectory(self, iceberg_id: str, model: str | None = None) -> dict | None:
        model = self._resolve_model(model)
        items = self._id_to_record()
        item = items.get(iceberg_id)
        if item is None:
            return None
        observations = self._observations_for(iceberg_id)
        # If the feature table had no rows for this iceberg, build one from the
        # latest position so the endpoint always returns at least one observation.
        if not observations:
            observations = [
                {
                    "timestamp": _parse_ts(item.get("last_observed")),
                    "step": "observation",
                    "latitude": item["latitude"],
                    "longitude": item["longitude"],
                }
            ]

        predictions: list[dict] = []
        if model in TRAINED_MODELS:
            out = self._trained_predict(iceberg_id, 24, model)
            if out is not None:
                for p in out["track"]:
                    predictions.append(
                        {
                            "timestamp": p.get("time"),
                            "step": "prediction",
                            "latitude": p["lat"],
                            "longitude": p["lon"],
                            "horizon_hours": round(p.get("hours_ahead") or 0, 1),
                        }
                    )
        if not predictions:
            predictor = self._predictor("persistence")
            predicted_raw = predictor.predict([item], horizon_hours=24)
            predictions = [
                {
                    "timestamp": None,
                    "step": "prediction",
                    "latitude": p.get("predicted_lat"),
                    "longitude": p.get("predicted_lon"),
                    "horizon_hours": p.get("prediction_hours", 24),
                }
                for p in predicted_raw
            ]
        return {
            "iceberg_id": iceberg_id,
            "observations": observations,
            "predictions": predictions,
            "count_observations": len(observations),
            "count_predictions": len(predictions),
            "prediction_model": model,
            **self._base(),
        }

    def model_status(self) -> dict:
        """Trained iceberg model availability (for the UI)."""
        return {
            "pipeline": "iceberg",
            "models": [
                get_runtime(m).summary
                for m in TRAINED_MODELS
            ],
        }

    def distance(self, iceberg_a: str, iceberg_b: str) -> dict | None:
        items = self._id_to_record()
        a = items.get(iceberg_a)
        b = items.get(iceberg_b)
        if a is None or b is None:
            return None
        dist = distance_between_icebergs(a["latitude"], a["longitude"], b["latitude"], b["longitude"])
        return {
            "iceberg_a": iceberg_a,
            "iceberg_b": iceberg_b,
            **dist,
            **self._base(),
        }


iceberg_service = IcebergService()