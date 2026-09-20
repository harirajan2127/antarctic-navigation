"""Sea-ice concentration forecasting service.

Provides a uniform interface over the forecasting models (persistence,
Random Forest, ConvLSTM). The model itself is selected from configuration.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.ml.sea_ice.persistence import PersistenceForecaster
from app.ml.sea_ice.random_forest import RandomForestIceForecaster
from app.ml.sea_ice.convlstm import ConvLSTMForecaster

logger = logging.getLogger(__name__)

SUPPORTED_HORIZONS_HOURS = [6, 12, 24, 48, 72]


class SeaIceForecastService:
    """Coordinates sea-ice data loading and model-based forecasting."""

    def __init__(self, model_name: str = "persistence") -> None:
        self.model_name = model_name
        self._model = self._build_model(model_name)

    @staticmethod
    def _build_model(model_name: str) -> Any:
        """Instantiate the requested forecasting model."""
        if model_name == "persistence":
            return PersistenceForecaster()
        if model_name == "random_forest":
            return RandomForestIceForecaster()
        if model_name == "convlstm":
            return ConvLSTMForecaster()
        raise ValueError(f"Unsupported sea-ice model: {model_name}")

    def forecast(
        self,
        concentration: list[list[float]],
        lat: list[float],
        lon: list[float],
        timestamp: datetime | None = None,
        horizon_hours: int = 24,
    ) -> dict[str, Any]:
        """Produce a sea-ice concentration forecast for the requested horizon.

        Returns a dictionary describing the forecast field, the model used,
        and the horizon. Real skill is only claimed when the model has been
        trained on real data.
        """
        if horizon_hours not in SUPPORTED_HORIZONS_HOURS:
            logger.warning(
                "Requested horizon %s hours not in supported set %s. Will still attempt.",
                horizon_hours,
                SUPPORTED_HORIZONS_HOURS,
            )

        forecast_field = self._json_safe_grid(
            self._model.predict(concentration, horizon_hours=horizon_hours)
        )
        ts = timestamp or datetime.now(timezone.utc)

        return {
            "forecast_time": ts.isoformat(),
            "horizon_hours": horizon_hours,
            "model": self.model_name,
            "lat": lat,
            "lon": lon,
            "concentration": forecast_field,
            "skill_note": self._model.skill_note(),
        }

    @staticmethod
    def _json_safe_grid(conc: list[list[float]]) -> list[list[float | None]]:
        """Replace NaN/Inf cells with None so the field serializes to JSON null
        (honest 'no data' marker — we never fabricate a concentration)."""
        import math

        return [
            [None if not math.isfinite(float(v)) else round(float(v), 6) for v in row]
            for row in conc
        ]

    def available_models(self) -> list[dict[str, str]]:
        """List forecasting models available in the system."""
        return [
            {
                "name": "persistence",
                "class": "PersistenceForecaster",
                "type": "baseline",
                "trained": str(PersistenceForecaster().is_trained),
            },
            {
                "name": "random_forest",
                "class": "RandomForestIceForecaster",
                "type": "machine_learning",
                "trained": str(RandomForestIceForecaster().is_trained),
            },
            {
                "name": "convlstm",
                "class": "ConvLSTMForecaster",
                "type": "deep_learning",
                "trained": str(ConvLSTMForecaster().is_trained),
            },
        ]