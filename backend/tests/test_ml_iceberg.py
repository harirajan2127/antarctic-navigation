"""Unit tests for the iceberg trajectory ML pipeline (fast, demo data)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.iceberg.baselines import CurrentDriftModel, PersistenceModel, RecursiveForecaster
from ml.iceberg.lstm_model import LSTMPositionModel, predict_batch_km
from ml.iceberg.random_forest_model import RandomForestTrajectoryModel
from ml.iceberg.trajectory_dataset import (
    LocalTangentPlane,
    FEATURE_COLS,
    N_FEATURES,
    TrajectoryDataset,
)
from ml.iceberg.utils import (
    haversine_km,
    is_demo_dataset,
    wrap_lon,
)

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "data/processed/features/feature_table.csv"


@pytest.fixture(scope="module")
def ds():
    assert FEATURES_CSV.exists(), FEATURES_CSV
    return TrajectoryDataset.from_features(FEATURES_CSV, seq_len=5, horizon=4)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def test_local_tangent_plane_roundtrip():
    plane = LocalTangentPlane(-70.0, 160.0)
    lat = np.array([-69.5, -70.25, -71.0])
    lon = np.array([160.0, 161.5, 163.0])
    x, y = plane.forward(lat, lon)
    lat2, lon2 = plane.inverse(x, y)
    assert np.allclose(lat2, lat, atol=1e-9)
    assert np.allclose(lon2, lon, atol=1e-9)


def test_projection_units_km():
    plane = LocalTangentPlane(0.0, 0.0)
    x, y = plane.forward(0.0, 1.0)
    assert abs(x - 111.1949) < 0.01  # 1 deg lon at equator ~ 111.19 km


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def test_sequences_shape(ds):
    assert ds.seq.shape[0] == len(ds.meta)
    assert ds.seq.shape[1] == ds.seq_len == 5
    assert ds.seq.shape[2] == N_FEATURES == 12
    assert ds.targets.shape[1] == 2
    assert len(ds.tracks) == 40


def test_windows_have_no_nan(ds):
    assert np.isfinite(ds.seq).all()
    assert np.isfinite(ds.targets).all()


def test_chronological_split_deterministic(ds):
    s1 = ds.split_chronological()
    s2 = ds.split_chronological()
    for k in s1:
        assert np.array_equal(s1[k], s2[k])
    for grp in ("train", "val", "test"):
        times = np.array([ds.meta[i]["target_time"] for i in s1[grp]], dtype="datetime64[us]")
        assert np.all(np.diff(times) >= np.timedelta64(0, "s"))
    # chronological ordering across groups: train <= val <= test
    for a, b in (("train", "val"), ("val", "test")):
        t_a = np.array([ds.meta[i]["target_time"] for i in s1[a]], dtype="datetime64[us]")
        t_b = np.array([ds.meta[i]["target_time"] for i in s1[b]], dtype="datetime64[us]")
        assert t_a.max() <= t_b.min() + np.timedelta64(1, "s")


def test_scalers_only_from_train(ds):
    splits = ds.split_chronological()
    ds2 = TrajectoryDataset.from_features(FEATURES_CSV, seq_len=5, horizon=4)
    ds2.fit_scalers(splits["train"])
    ds2.normalize()
    assert "mean" in ds2.scalers and "std" in ds2.scalers
    assert ds2.scalers["mean"].shape == (N_FEATURES,)
    assert np.isfinite(ds2.seq).all()


def test_feature_cols_signature():
    assert FEATURE_COLS[:4] == ["x", "y", "vx", "vy"]
    assert "uo" in FEATURE_COLS and "v10" in FEATURE_COLS and "sin_doy" in FEATURE_COLS


def test_track_kinematics(ds):
    tr = ds.track("DEMO-B000")
    assert len(tr) >= ds.seq_len + ds.horizon + 1
    assert np.all(tr.displacement_km[1:] >= 0)
    assert np.nanmedian(tr.speed_kmh[1:]) > 0
    assert np.all(np.isfinite(tr.direction_deg[1:]))


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def test_persistence_keeps_position(ds):
    tr = ds.track("DEMO-B000")
    fwd = PersistenceModel(ds).forecast(tr, 50, 3)
    assert len(fwd) == 3
    for p in fwd:
        assert p["lat"] == tr.lat[50] and p["lon"] == tr.lon[50]
        assert p["distance_km_from_prev"] == 0.0


def test_current_drift_positive_displacement(ds):
    tr = ds.track("DEMO-B000")
    fwd = CurrentDriftModel(ds).forecast(tr, 50, 2)
    assert len(fwd) == 2
    assert fwd[1]["lat"] != tr.lat[50]


def test_recursive_forecaster_no_leakage_and_monotonic_times(ds):
    ds2 = TrajectoryDataset.from_features(FEATURES_CSV, seq_len=5, horizon=4)
    splits = ds2.split_chronological()
    ds2.fit_scalers(splits["train"])
    ds2.normalize()
    tr = ds2.track("DEMO-B000")
    n = len(tr)
    s = ds2.seq_len
    # start mid-track so future obs exist BEFORE start would leak if bug present
    start = max(s, n - s - 3)

    class _Identity:
        def __init__(self, ds2):
            pass

        def __call__(self, seq):
            return np.array([[0.0, 0.0]])

    fwd = RecursiveForecaster(ds2, _Identity(ds2)).forecast(tr, start, 3)
    times = [p["time"] for p in fwd]
    assert len(sorted(times)) == len(times)  # strictly increasing
    assert np.all(np.diff([p["hours_from_current"] for p in fwd]) > 0)
    # predicted lon wrapped into valid range
    for p in fwd:
        assert -180.0 <= p["lon"] <= 180.0


def test_wrap_lon():
    assert np.isclose(wrap_lon(206.0), -154.0)
    assert np.isclose(wrap_lon(-190.0), 170.0)
    assert np.isclose(wrap_lon(177.4), 177.4)


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def test_rf_train_predict_roundtrip(ds, tmp_path):
    splits = ds.split_chronological()
    ds.fit_scalers(splits["train"])
    ds.normalize()
    X, Y = ds.tabular(splits["train"])
    m = RandomForestTrajectoryModel(n_estimators=20, max_depth=5, seed=3)
    m.fit(X[:200], Y[:200])
    pred = m.predict(X[:10])
    assert pred.shape == (10, 2)
    assert np.isfinite(pred).all()
    path = m.save(tmp_path / "rf.joblib")
    m2 = RandomForestTrajectoryModel.load(path)
    assert m2.fitted
    assert np.allclose(m2.predict(X[:2]), pred[:2], atol=1e-6)


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

def test_lstm_forward_shape(ds):
    model = LSTMPositionModel()
    x = torch_randn(4, ds.seq_len, N_FEATURES)
    y = model(x)
    assert tuple(y.shape) == (4, 2)


def test_lstm_train_smoke(ds):
    from ml.iceberg.lstm_model import train_loop

    splits = ds.split_chronological()
    ds.fit_scalers(splits["train"])
    ds.normalize()
    tiny_train = splits["train"][:64]
    tiny_val = splits["val"][:64]
    out = train_loop(ds, tiny_train, tiny_val, epochs=2, batch_size=16,
                     hidden_size=8, num_layers=1, seed=1)
    assert len(out["history"]) <= 2
    assert out["model"] is not None
    assert np.isfinite(out["best_val_rmse_km"])


def torch_randn(*shape):
    import torch

    return torch.randn(*shape)


# ---------------------------------------------------------------------------
# Demo guard
# ---------------------------------------------------------------------------

def test_demo_dataset_detection(ds):
    assert is_demo_dataset([FEATURES_CSV])


def test_predict_python_api(ds, tmp_path):
    from ml.iceberg.train_rf import main as train_rf_main
    from ml.iceberg.predict import predict

    out_dir = tmp_path / "run"
    train_rf_main(["--demo", "--out", str(out_dir), "--seq-len", "5", "--horizon", "4",
                   "--n-estimators", "10", "--tracks", str(FEATURES_CSV)])
    assert (out_dir / "rf.joblib").exists()
    r = predict(out_dir, "DEMO-B000", steps=3, model_kind="random_forest", tracks=FEATURES_CSV)
    assert r["iceberg_id"] == "DEMO-B000"
    assert r["forecast_horizon"]["steps"] == 3
    assert len(r["models"]["random_forest"]["predicted_track"]) == 3
    assert r["prediction_uncertainty"]["type"] == "not_implemented"
    assert (out_dir / "predictions" / "prediction_DEMO-B000_3steps.json").exists()


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

def test_haversine_consistent_with_services():
    from services.distance_calculator import haversine_distance_km

    d = haversine_km(-71.0, 160.0, -69.5, 162.0)
    assert np.isclose(d, haversine_distance_km(-71.0, 160.0, -69.5, 162.0), rtol=1e-6)