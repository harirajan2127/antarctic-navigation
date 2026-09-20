"""Random Forest regression for iceberg trajectory prediction (skeleton)."""
from __future__ import annotations

import numpy as np


class RandomForestIcebergPredictor:
    """Random Forest regression predicting east/west displacement.

    Features per iceberg: current lat, lon, size (km2), drift speed, history
    window. Targets: dlat and dlon at the forecast horizon.
    """

    def __init__(self) -> None:
        self.models: dict[int, dict[str, object]] = {}
        self.is_trained = False
        self.metric_note = (
            "Not trained. Metrics will be reported after training on "
            "validated iceberg track data."
        )

    def fit(
        self,
        X: np.ndarray,
        y_lat: np.ndarray,
        y_lon: np.ndarray,
        horizon_hours: int,
    ) -> None:
        """Fit random forests for dlat and dlon displacement."""
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError as exc:
            raise RuntimeError(
                "scikit-learn is required to train the Random Forest iceberg predictor."
            ) from exc

        model_lat = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
        model_lon = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
        model_lat.fit(X, y_lat)
        model_lon.fit(X, y_lon)
        self.models[horizon_hours] = {"lat": model_lat, "lon": model_lon}
        self.is_trained = True

    def predict_position(self, iceberg: dict, horizon_hours: int):
        """Predict position. Falls back to persistence when untrained."""
        if not self.is_trained or horizon_hours not in self.models:
            # Persistence fallback.
            lat = iceberg.get("latitude", 0.0)
            lon = iceberg.get("longitude", 0.0)
            speed = iceberg.get("drift_speed_km_per_hour", 1.0)
            direction = iceberg.get("drift_direction_deg", 225.0)
            from app.ml.iceberg.persistence import PersistenceIcebergPredictor

            return PersistenceIcebergPredictor()._destination_point(
                lat, lon, direction, speed * horizon_hours
            )

        raise NotImplementedError(
            "Learned inference is implemented in Phase 3 when training data is available."
        )

    def confidence_note(self) -> str:
        return self.metric_note