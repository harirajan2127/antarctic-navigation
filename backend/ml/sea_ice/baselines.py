"""Baseline models for sea-ice forecasting.

* Model A - persistence: ``tomorrow = today``.
* Model B - Random Forest on tabular (sample, cell) features.

Evaluation numerics come from the same masked metric functions as the
primary model, so all comparisons are apples-to-apples.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .dataset import SeaIceDataset, TARGET_CHANNEL
from .utils import (
    masked_mae,
    masked_rmse,
    masked_spatial_correlation,
)

LOG = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    name: str
    mae: float
    rmse: float
    spatial_correlation: float
    n_valid_cells: int
    prediction: np.ndarray | None = None  # [n, H, W] on the requested split


def evaluate_persistence(
    ds: SeaIceDataset, idx: np.ndarray
) -> BaselineResult:
    """Model A prediction grid: last observed SIC channel, clipped to [0, 1]."""
    pred = ds.X_raw[idx][:, -1, :, :, TARGET_CHANNEL].copy()
    pred = np.clip(pred, 0.0, 1.0)
    y = ds.y[idx]
    mask = ds.mask[idx]
    return BaselineResult(
        name="persistence",
        mae=masked_mae(pred, y, mask),
        rmse=masked_rmse(pred, y, mask),
        spatial_correlation=masked_spatial_correlation(pred, y, mask),
        n_valid_cells=int(mask.sum()),
        prediction=pred,
    )


def fit_random_forest(
    ds: SeaIceDataset,
    train_idx: np.ndarray,
    n_estimators: int = 100,
    max_depth: int | None = None,
    subsample: int | None = 200_000,
    random_state: int = 0,
) -> RandomForestRegressor:
    """Train Model B on flattened (sample, cell) features from the *train* split."""
    x_train, y_train, m_train = ds.tabular_cells(train_idx)
    ok = m_train.astype(bool)
    if ok.sum() == 0:
        raise ValueError("Random forest: no valid training cells available.")
    if subsample is not None and ok.sum() > subsample:
        rng = np.random.default_rng(random_state)
        pick = rng.choice(ok.sum(), size=subsample, replace=False)
        x_ok, y_ok = x_train[ok], y_train[ok]
        x_train, y_train = x_ok[pick], y_ok[pick]
    else:
        x_train, y_train = x_train[ok], y_train[ok]
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state,
    )
    LOG.info("Training random forest on %d rows x %d cols ...", *x_train.shape)
    rf.fit(x_train, y_train)
    return rf


def evaluate_random_forest(
    rf: RandomForestRegressor, ds: SeaIceDataset, idx: np.ndarray
) -> BaselineResult:
    """Score Model B on a split, reconstructing per-sample grids."""
    x_all, _, m_all = ds.tabular_cells(idx)
    pred_all = np.clip(rf.predict(x_all), 0.0, 1.0)
    h, w = ds.lats.size, ds.lons.size
    n = x_all.shape[0] // (h * w)
    pred_grid = pred_all.reshape(n, h, w).copy()
    pred_grid[~ds.mask[idx]] = np.nan

    y = ds.y[idx]
    mask = ds.mask[idx]
    return BaselineResult(
        name="random_forest",
        mae=masked_mae(pred_grid, y, mask),
        rmse=masked_rmse(pred_grid, y, mask),
        spatial_correlation=masked_spatial_correlation(pred_grid, y, mask),
        n_valid_cells=int(mask.sum()),
        prediction=pred_grid,
    )


def save_rf(rf: RandomForestRegressor, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, path)
    return path


def load_rf(path: Path | str) -> RandomForestRegressor:
    return joblib.load(Path(path))