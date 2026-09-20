"""Random Forest baseline sea-ice forecaster (skeleton).

Implements the training pipeline interface. Real training requires a
labeled historical dataset of sea-ice concentration fields. No real model
metrics are reported until a model has been trained on real data.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class RandomForestIceForecaster:
    """Random Forest regression on cell-wise features.

    Features (per grid cell): latitude, longitude, current-day
    concentration, day-of-year. Target: concentration at the forecast
    horizon. A single forest is trained for each supported horizon.
    """

    def __init__(self, n_estimators: int = 100) -> None:
        self.n_estimators = n_estimators
        self.models: dict[int, Any] = {}
        self.is_trained = False
        self.metric_note = (
            "Not trained. Metrics will be reported after training on a "
            "real, validated sea-ice dataset."
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        horizon_hours: int,
    ) -> None:
        """Fit a RandomForestRegressor for a given forecast horizon."""
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError as exc:
            raise RuntimeError(
                "scikit-learn is required to train the Random Forest ice forecaster."
            ) from exc

        model = RandomForestRegressor(
            n_estimators=self.n_estimators, max_depth=12, random_state=42, n_jobs=-1
        )
        model.fit(X, y)
        self.models[horizon_hours] = model
        self.is_trained = True

    def predict(self, concentration: list[list[float]], horizon_hours: int) -> list[list[float]]:
        """Predict concentration at the horizon using the trained forest.

        Falls back to persistence with a warning when no model is trained.
        """
        if not self.is_trained or horizon_hours not in self.models:
            return [list(map(float, row)) for row in concentration]

        raise NotImplementedError(
            "Feature construction from grid fields is implemented in Phase 3."
        )

    def skill_note(self) -> str:
        return self.metric_note