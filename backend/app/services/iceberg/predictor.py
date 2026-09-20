"""Iceberg trajectory prediction service."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from app.ml.iceberg.persistence import PersistenceIcebergPredictor
from app.ml.iceberg.random_forest import RandomForestIcebergPredictor
from app.ml.iceberg.lstm import LSTMIcebergPredictor

logger = logging.getLogger(__name__)

SUPPORTED_HORIZONS_HOURS = [2, 4, 6, 12, 24, 48, 72]


class IcebergTrajectoryService:
    """Coordinates iceberg data loading and trajectory prediction."""

    def __init__(self, model_name: str = "persistence") -> None:
        self.model_name = model_name
        self._predictor = self._build_predictor(model_name)

    @staticmethod
    def _build_predictor(model_name: str) -> Any:
        if model_name == "persistence":
            return PersistenceIcebergPredictor()
        if model_name == "random_forest":
            return RandomForestIcebergPredictor()
        if model_name == "lstm":
            return LSTMIcebergPredictor()
        raise ValueError(f"Unsupported iceberg model: {model_name}")

    def predict(
        self,
        icebergs: list[dict[str, Any]],
        horizon_hours: int = 24,
    ) -> list[dict[str, Any]]:
        """Predict iceberg positions at a given horizon.

        Returns predicted positions for each iceberg along with a confidence
        value. Confidence should only be derived from real model validation,
        never invented.
        """
        if horizon_hours not in SUPPORTED_HORIZONS_HOURS:
            logger.warning("Requested horizon %s not in supported set.", horizon_hours)

        predictions = []
        for iceberg in icebergs:
            lat, lon = self._predictor.predict_position(
                iceberg,
                horizon_hours=horizon_hours,
            )
            predictions.append(
                {
                    "iceberg_id": iceberg["iceberg_id"],
                    "current_lat": iceberg["latitude"],
                    "current_lon": iceberg["longitude"],
                    "predicted_lat": round(float(lat), 4),
                    "predicted_lon": round(float(lon), 4),
                    "prediction_hours": horizon_hours,
                    "confidence": self._predictor.confidence_note(),
                }
            )
        return predictions

    def predict_multi_horizon(
        self, icebergs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Predict iceberg positions across all supported horizons."""
        result = []
        for h in SUPPORTED_HORIZONS_HOURS:
            for p in self.predict(icebergs, horizon_hours=h):
                result.append(p)
        return result

    def available_models(self) -> list[dict[str, str]]:
        """List iceberg prediction models available in the system."""
        return [
            {
                "name": "persistence",
                "class": "PersistenceIcebergPredictor",
                "type": "baseline",
                "trained": str(PersistenceIcebergPredictor().is_trained),
            },
            {
                "name": "random_forest",
                "class": "RandomForestIcebergPredictor",
                "type": "machine_learning",
                "trained": str(RandomForestIcebergPredictor().is_trained),
            },
            {
                "name": "lstm",
                "class": "LSTMIcebergPredictor",
                "type": "deep_learning",
                "trained": str(LSTMIcebergPredictor().is_trained),
            },
        ]