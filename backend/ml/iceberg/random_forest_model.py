"""Random Forest regression for iceberg next-position forecasting.

One independent Random Forest per target axis (projected X and Y in km). This
keeps both models embarrassingly parallel and avoids the correlated-target
stacking of ``MultiOutputRegressor``.

Inputs (tree input features):
    past window flattened  [seq_len x 12]:
        x, y, vx, vy, uo, vo, u10, v10, length, width, sin_doy, cos_doy
Targets:
    future projected X, Y (km) — the co-ordinates are inverted to lat/lon by
    the caller using the sequence's local tangent plane origin.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

LOG = logging.getLogger(__name__)


class RandomForestTrajectoryModel:
    name = "random_forest"

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int | None = 20,
        min_samples_leaf: int = 3,
        n_jobs: int = -1,
        seed: int = 7,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
        self.models = {
            x_axis: RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                n_jobs=n_jobs,
                random_state=seed,
            )
            for x_axis in ("x", "y")
        }
        self.fitted = False

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, Y: np.ndarray) -> "RandomForestTrajectoryModel":
        """X [N, features], Y [N, 2] projected km."""
        for axis, m in self.models.items():
            self._fit_axis(m, axis, X, Y)
        self.fitted = True
        return self

    def _fit_axis(self, model, axis, X, Y):
        idx = int(axis == "y")
        model.fit(X, Y[:, idx])
        LOG.info("RF[%s] trained (n=%d, features=%d)", axis, len(X), X.shape[1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X [N, features] -> [N, 2] projected km."""
        return np.column_stack([self.models[k].predict(X) for k in ("x", "y")])

    # ------------------------------------------------------------------
    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self,
                "fitted": self.fitted,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            },
            path,
        )
        LOG.info("Saved Random Forest model -> %s", path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "RandomForestTrajectoryModel":
        obj = joblib.load(path)
        if isinstance(obj, dict):
            model = obj["model"]
        else:
            model = obj
        if not getattr(model, "fitted", False):
            raise ValueError(f"Model at {path} was never fitted.")
        return model