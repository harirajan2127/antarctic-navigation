"""Unit tests for the sea-ice ML forecasting pipeline (fast, small grids)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ml.sea_ice.baselines import evaluate_persistence, fit_random_forest
from ml.sea_ice.dataset import (
    ALL_CHANNELS,
    N_CHANNELS,
    SeaIceDataset,
    SeaIceWindowDataset,
)
from ml.sea_ice.model import ConvLSTM, PersistenceModel, build_model, count_parameters
from ml.sea_ice.utils import (
    fit_channel_scaler,
    is_demo_dataset,
    masked_mae,
    masked_rmse,
    masked_spatial_correlation,
    set_seed,
)

DEMO = Path(__file__).resolve().parents[1] / "data"
SEA_ICE = DEMO / "processed/sea_ice/sea_ice_clean.nc"
OCEAN = DEMO / "processed/ocean/ocean_clean.nc"
WEATHER = DEMO / "processed/weather/weather_clean.nc"


@pytest.fixture(scope="module")
def demo_files():
    for p in (SEA_ICE, OCEAN, WEATHER):
        assert p.exists(), p
    return {"sea_ice": SEA_ICE, "ocean": OCEAN, "weather": WEATHER}


@pytest.fixture(scope="module")
def ds(demo_files):
    return SeaIceDataset.from_files(demo_files, input_steps=7, horizon=1, spatial_step=20)


# ---------------------------------------------------------------------------
# Masked metrics
# ---------------------------------------------------------------------------

def test_masked_mae_rmse_hand():
    pred = np.array([[0.6, 0.7], [0.8, np.nan]])
    actual = np.array([[0.5, 1.0], [0.7, 0.4]])
    mask = np.array([[True, True], [True, False]])
    # valid cells: (0,0),(0,1),(1,0) -> errors 0.1, 0.3, 0.1
    assert np.isclose(masked_mae(pred, actual, mask), 0.1666667, atol=1e-4)
    assert np.isclose(masked_rmse(pred, actual, mask), 0.1914854, atol=1e-4)


def test_masked_metrics_drop_nan_pred():
    pred = np.array([[np.nan, 0.7]])
    actual = np.array([[0.5, 1.0]])
    mask = np.array([[True, True]])
    assert np.isclose(masked_mae(pred, actual, mask), 0.3)


def test_masked_spatial_correlation_perfect():
    rng = np.random.default_rng(0)
    a = rng.normal(size=50)
    p = a + 0.01 * rng.normal(size=50)
    r = masked_spatial_correlation(p[None], a[None], np.ones((1, 50), dtype=bool))
    assert np.isclose(r, 1.0, atol=1e-4)


def test_spatial_correlation_nan_on_degenerate():
    r = masked_spatial_correlation(np.ones((1, 5)), np.ones((1, 5)), np.ones((1, 5), bool))
    assert np.isnan(r)


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def test_fit_channel_scaler_ignores_nan_cells():
    x = np.array([[[1.0], [2.0]], [[3.0], [np.nan]]])  # single channel, 1 NaN of 4
    mask = np.ones((2, 2), dtype=bool)
    stats = fit_channel_scaler(x, mask)
    finite = np.array([1.0, 2.0, 3.0])
    assert np.isclose(stats["mean"][0], finite.mean())
    assert np.isclose(stats["std"][0], finite.std())


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def test_dataset_shapes(ds):
    assert ds.X.shape == (8, 7, 2, 18, N_CHANNELS)
    assert ds.y.shape == (8, 2, 18)
    assert ds.mask.shape == (8, 2, 18)
    assert ds.X.dtype == np.float32


def test_dataset_time_window_alignment(ds):
    # target dates are one day apart (daily grid, horizon=1, input_steps=7)
    assert len(ds.times) == 8
    assert (ds.times.diff().dropna() == pd.Timedelta(days=1)).all()


def test_chronological_split_contiguous_and_covering(ds):
    sp = ds.split_chronological(0.7, 0.15)
    idx = np.concatenate([sp["train"], sp["val"], sp["test"]])
    assert np.array_equal(idx, np.arange(ds.X.shape[0]))
    # strictly increasing time
    times = ds.times[idx]
    assert (times.diff().dropna() > pd.Timedelta(0)).all()


def test_mask_only_finite_windows(ds):
    raw = np.isfinite(ds.X_raw)  # [n, Ti, H, W, F]
    assert np.all(ds.mask == (np.all(raw, axis=(1, 4)) & np.isfinite(ds.y)))


def test_normalize_fills_nan_with_zero(ds):
    ds.fit_scalers(np.arange(6))
    ds.normalize()
    assert not np.isnan(ds.X).any()


def test_tabular_cells_ordering(ds):
    x, y, m = ds.tabular_cells(np.array([3, 4]))
    h, w = ds.lats.size, ds.lons.size
    assert x.shape == (2 * h * w, 7 * N_CHANNELS)
    np.testing.assert_allclose(x[0], ds.X[3, :, 0, 0, :].reshape(7 * N_CHANNELS))
    np.testing.assert_allclose(x[h * w], ds.X[4, :, 0, 0, :].reshape(7 * N_CHANNELS))


def test_window_dataset(ds):
    w = SeaIceWindowDataset(ds, np.array([0, 1]))
    assert len(w) == 2
    item = w[0]
    assert tuple(item["x"].shape) == (7, ds.lats.size, ds.lons.size, N_CHANNELS)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def test_convlstm_forward_shape():
    set_seed(0)
    m = ConvLSTM(in_channels=5, hidden_channels=8)
    x = torch.randn(2, 7, 12, 13, 5)
    out = m(x)
    assert tuple(out.shape) == (2, 12, 13, 1)


def test_convlstm_predict_grid_bounded():
    set_seed(0)
    m = ConvLSTM(in_channels=5, hidden_channels=8).eval()
    x = torch.randn(1, 7, 8, 9, 5)
    out = m.predict_grid(x)
    assert (out >= 0).all() and (out <= 1).all()


def test_persistence_model_forward():
    m = PersistenceModel()
    x = torch.randn(3, 7, 5, 6, 11)
    out = m(x)
    assert tuple(out.shape) == (3, 5, 6, 1)
    torch.testing.assert_close(out[..., 0], x[:, -1, :, :, 0])


def test_build_model_and_params():
    m = build_model("convlstm", {"in_channels": 11, "hidden_channels": 8},
                    torch.device("cpu"))
    assert count_parameters(m) > 0


def test_dispatched_model_names():
    with pytest.raises(ValueError):
        build_model("nope", {"in_channels": 11}, torch.device("cpu"))


# ---------------------------------------------------------------------------
# Demo guard & baselines
# ---------------------------------------------------------------------------

def test_demo_detection_on_synthetic_inputs(demo_files):
    assert is_demo_dataset(list(demo_files.values())) is True


def test_persistence_and_rf_on_split(ds):
    from ml.sea_ice.baselines import evaluate_random_forest

    sp = ds.split_chronological()
    pers = evaluate_persistence(ds, sp["test"])
    assert np.isfinite(pers.mae) and np.isfinite(pers.rmse)
    ds.fit_scalers(sp["train"])
    ds.normalize()
    rf = fit_random_forest(ds, sp["train"], n_estimators=10, subsample=5000)
    r = evaluate_random_forest(rf, ds, sp["test"])
    assert np.isfinite(r.mae) and np.isfinite(r.rmse)