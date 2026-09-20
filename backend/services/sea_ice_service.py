"""Sea-ice data and forecasting service.

Wraps ``app.services.sea_ice`` behind a stable demo-aware interface. The
sea-ice loader is instantiated once at module level and cached — data files
and demo fields are never re-read from disk per request.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.services.sea_ice.data_loader import SeaIceDataLoader
from app.services.sea_ice.forecaster import SeaIceForecastService
from config import DEMO_WARNING, effective_demo, settings
from database.database import SessionLocal
from database.models import SeaIceForecastRecord
from ml.sea_ice.runtime import get_runtime, SeaIceRuntime
from services.data_paths import SEA_ICE_NETCDF

logger = logging.getLogger("dss.services.sea_ice")

TRAINED_MODELS = ("convlstm", "random_forest")


def _display_classification(classification: str) -> str:
    """Visible label: real processed data shows as ``real_sea_ice``."""
    if classification == "pipeline_data":
        return "real_sea_ice"
    return classification


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class SeaIceService:
    """Thread-safe, lazily-cached sea-ice service (one instance per process)."""

    def __init__(self) -> None:
        self._loader = SeaIceDataLoader(netcdf_path=str(SEA_ICE_NETCDF))
        self._cache: dict = {}
        self._model_cache: dict[str, SeaIceForecastService] = {}

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _load(self) -> dict:
        if not self._cache:
            self._cache = self._loader.load()
        return self._cache

    def _forecaster(self, model: str) -> SeaIceForecastService:
        if model not in self._model_cache:
            self._model_cache[model] = SeaIceForecastService(model_name=model)
        return self._model_cache[model]

    def _resolve_model(self, model: str | None) -> str:
        """Fall back to the configured model when the caller passes nothing."""
        if not model or model == "auto":
            chosen = (settings.SEA_ICE_MODEL or "persistence").strip().lower()
            return chosen if chosen else "persistence"
        return model.strip().lower()

    def _trained_grid(self, model: str, horizon_hours: int) -> tuple[dict | None, SeaIceRuntime | None]:
        """Return ``(grid_dict, runtime)`` from a trained model, if usable.

        Only the model's native horizon (24 h for the current trained runs) is
        served from the runtime; otherwise ``(None, runtime)`` so the caller can
        fall back to persistence with a clear note.
        """
        if model not in TRAINED_MODELS:
            return None, None
        runtime = get_runtime(model)
        if not runtime.available or runtime.error:
            return None, runtime
        if horizon_hours not in (24,):
            return None, runtime
        try:
            return runtime.grid_predict(), runtime
        except Exception:  # noqa: BLE001 - never crash on a bad checkpoint
            logger.exception("Trained sea-ice runtime '%s' failed at inference", model)
            return None, runtime

    def _common(self, data: dict) -> dict:
        raw_cls = str(data.get("classification", "unknown"))
        demo = effective_demo(raw_cls)
        return {
            "lat": data.get("lat", []),
            "lon": data.get("lon", []),
            "classification": _display_classification(raw_cls),
            "demo": demo,
            "source": data.get("path"),
            "warning": data.get("demo_notice") or (DEMO_WARNING if demo else None),
        }

    def _now_and_valid(self, iso_time: str | None, horizon: int) -> tuple[datetime, datetime]:
        from datetime import timedelta

        try:
            base = datetime.fromisoformat(str(iso_time).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            base = _utcnow()
        if base.tzinfo is not None:
            base = base.astimezone(timezone.utc).replace(tzinfo=None)
        return base, base + timedelta(hours=horizon)

    def _summary(self, conc: list[list[float]], lat: list[float], lon: list[float]) -> dict:
        import math
        flat = [v for row in conc for v in row]
        valid = [v for v in flat if v is not None and isinstance(v, (int, float)) and math.isfinite(v)]
        if not valid:
            return {"mean": 0.0, "max": 0.0, "coverage": 0.0}
        valid_total = sum(1 for v in flat if v is not None and isinstance(v, (int, float)) and math.isfinite(v))
        valid_pct = round(100.0 * valid_total / len(flat), 2) if flat else 0.0
        return {
            "mean": round(sum(valid) / len(valid), 6),
            "max": round(max(valid), 6),
            "coverage": valid_pct,
        }

    @staticmethod
    def _json_safe_grid(conc: list[list[float]]) -> list[list[float | None]]:
        """Replace NaN/Inf cells with None so the field serializes to JSON null
        (honest 'no data' marker — we never fabricate a concentration)."""
        import math

        def _cell(v: float | None) -> float | None:
            if v is None:
                return None
            f = float(v)
            return None if not math.isfinite(f) else round(f, 6)

        return [[_cell(v) for v in row] for row in conc]

    def _persist_forecast(self, result: dict, model: str, horizon: int) -> None:
        valid_time = result.get("valid_time")
        summary = self._summary(
            result.get("concentration", []), result.get("lat", []), result.get("lon", [])
        )
        demo = result.get("demo", True)
        with SessionLocal() as db:
            db.add(SeaIceForecastRecord(
                forecast_time=result.get("forecast_time", _utcnow()),
                valid_time=valid_time,
                horizon_hours=horizon,
                model=model,
                mean_concentration=summary["mean"],
                max_concentration=summary["max"],
                coverage_pct=summary["coverage"],
                demo=demo,
            ))
            db.commit()

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------
    def current(self) -> dict:
        """Current sea-ice concentration field (one time step)."""
        data = self._load()
        out = self._common(data)
        out.update({
            "timestamp": _utcnow(),
            "concentration": self._json_safe_grid(data.get("sea_ice_concentration", [])),
        })
        return out

    def forecast(
        self,
        horizon_hours: int = 24,
        model: str | None = None,
        timestamp: datetime | None = None,
    ) -> dict:
        """Produce a sea-ice concentration forecast.

        Uses the trained runtime (`convlstm` / `random_forest`) when artifacts
        exist and the horizon matches the model's native step; otherwise it
        falls back to the persistence baseline and says so in the skill note.
        """
        model = self._resolve_model(model)
        data = self._load()

        grid, runtime = self._trained_grid(model, horizon_hours)
        forecast_time = _utcnow()
        if grid is not None:
            conc = grid.get("concentration", [])
            lat = grid.get("lat", data.get("lat", []))
            lon = grid.get("lon", data.get("lon", []))
            summary = self._summary(conc, lat, lon)
            classification = data.get("classification", "unknown")
            demo = effective_demo(classification) or bool(grid.get("demo_only", True))
            _, valid_time = self._now_and_valid(None, horizon_hours)
            skill_note = grid.get("skill_note")
            if demo:
                skill_note = (skill_note or "") + " Run on synthetic demo data — no real skill."
            result = {
                "forecast_time": forecast_time,
                "valid_time": valid_time,
                "horizon_hours": horizon_hours,
                "model": model,
                "latitude": "tmp",
                "lat": lat,
                "lon": lon,
                "concentration": self._json_safe_grid(conc),
                "mean_concentration": summary["mean"],
                "max_concentration": summary["max"],
                "coverage_pct": summary["coverage"],
                "classification": _display_classification(classification),
                "demo": False,
                "model_used_real": True,
                "skill_note": skill_note,
                "warning": None,
            }
            self._persist_forecast(result, model, horizon_hours)
            return result

        forecaster = self._forecaster(model)
        forecast_result = forecaster.forecast(
            concentration=data.get("sea_ice_concentration", []),
            lat=data.get("lat", []),
            lon=data.get("lon", []),
            timestamp=timestamp,
            horizon_hours=horizon_hours,
        )
        conc = forecast_result.get("concentration", [])
        lat = data.get("lat", [])
        lon = data.get("lon", [])
        summary = self._summary(conc, lat, lon)
        classification = data.get("classification", "unknown")
        demo = effective_demo(classification)
        forecast_time, valid_time = self._now_and_valid(
            forecast_result.get("forecast_time"), horizon_hours
        )
        skill_note = forecast_result.get("skill_note")
        if runtime is not None and not runtime.available:
            skill_note = (skill_note or "") + " Trained model unavailable — persistence used."
        elif runtime is not None and runtime.available:
            skill_note = (skill_note or "") + " Persistence used (loaded model does not support this horizon)."
        result = {
            "forecast_time": forecast_time,
            "valid_time": valid_time,
            "horizon_hours": horizon_hours,
            "model": model,
            "lat": lat,
            "lon": lon,
            "concentration": self._json_safe_grid(conc),
            "mean_concentration": summary["mean"],
            "max_concentration": summary["max"],
            "coverage_pct": summary["coverage"],
            "classification": _display_classification(classification),
            "demo": demo,
            "model_used_real": not demo,
            "skill_note": skill_note,
            "warning": DEMO_WARNING if demo else None,
        }
        self._persist_forecast(result, model, horizon_hours)
        return result

    def model_status(self) -> dict:
        """Which trained sea-ice models are available on disk (for the UI)."""
        out: list[dict] = []
        for name in TRAINED_MODELS:
            r = get_runtime(name)
            out.append(
                {
                    "name": name,
                    "available": r.available,
                    "trained_on_demo": bool(r.cfg.get("demo", True)) if r.cfg else None,
                    "error": r.error,
                }
            )
        return {"pipeline": "sea_ice", "models": out}

    def predict(
        self,
        concentration: list[list[float]] | None = None,
        lat: list[float] | None = None,
        lon: list[float] | None = None,
        horizon_hours: int = 24,
        model: str | None = None,
        timestamp: datetime | None = None,
    ) -> dict:
        """Forecast sea-ice concentration from optional custom inputs."""
        model = self._resolve_model(model)
        data = self._load()
        conc = concentration if concentration is not None else data.get("sea_ice_concentration", [])
        lats = lat if lat is not None else data.get("lat", [])
        lons = lon if lon is not None else data.get("lon", [])
        forecaster = self._forecaster(model)
        forecast_result = forecaster.forecast(
            concentration=conc,
            lat=lats,
            lon=lons,
            timestamp=timestamp,
            horizon_hours=horizon_hours,
        )
        out_conc = forecast_result.get("concentration", [])
        summary = self._summary(out_conc, lats, lons)
        classification = data.get("classification", "unknown")
        demo = effective_demo(classification)
        forecast_time, valid_time = self._now_and_valid(
            forecast_result.get("forecast_time"), horizon_hours
        )
        result = {
            "forecast_time": forecast_time,
            "valid_time": valid_time,
            "horizon_hours": horizon_hours,
            "model": model,
            "lat": lats,
            "lon": lons,
            "concentration": self._json_safe_grid(out_conc),
            "mean_concentration": summary["mean"],
            "max_concentration": summary["max"],
            "coverage_pct": summary["coverage"],
            "classification": _display_classification(classification),
            "demo": demo,
            "model_used_real": not demo,
            "skill_note": forecast_result.get("skill_note"),
            "warning": DEMO_WARNING if demo else None,
        }
        self._persist_forecast(result, model, horizon_hours)
        return result

    def statistics(self) -> dict:
        """Return DB-level forecast statistics for analytics."""
        with SessionLocal() as db:
            rows = db.scalars(select(SeaIceForecastRecord)).all()
        if not rows:
            return {"total": 0, "by_model": {}, "demo_count": 0}
        by_model: dict[str, int] = {}
        demo_count = 0
        for r in rows:
            by_model[r.model] = by_model.get(r.model, 0) + 1
            if r.demo:
                demo_count += 1
        return {"total": len(rows), "by_model": by_model, "demo_count": demo_count}


sea_ice_service = SeaIceService()