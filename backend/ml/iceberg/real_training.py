from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger("dss.ml.iceberg.real_training")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_DATA = PROJECT_ROOT.parent / "Real data" / "processed" / "iceberg" / "csv" / "iceberg_processed.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "iceberg" / "run_real"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r1 = np.deg2rad(lat1)
    r2 = np.deg2rad(lat2)
    dlat = np.deg2rad(lat2 - lat1)
    dlon = np.deg2rad(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(r1) * np.cos(r2) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r1 = np.deg2rad(lat1)
    r2 = np.deg2rad(lat2)
    dlon = np.deg2rad(lon2 - lon1)
    y = np.sin(dlon) * np.cos(r2)
    x = np.cos(r1) * np.sin(r2) - np.sin(r1) * np.cos(r2) * np.cos(dlon)
    return float((np.rad2deg(np.arctan2(y, x)) + 360.0) % 360.0)


def season_for(month: int) -> str:
    if month in (12, 1, 2):
        return "summer"
    if month in (3, 4, 5):
        return "autumn"
    if month in (6, 7, 8):
        return "winter"
    return "spring"


def load_and_clean() -> pd.DataFrame:
    if not REAL_DATA.exists():
        raise FileNotFoundError(f"Real iceberg data not found: {REAL_DATA}")

    df = pd.read_csv(REAL_DATA)
    n_before = len(df)
    LOGGER.info("ICEBERG rows before cleaning: %d", n_before)

    df = df.copy()
    df["iceberg_id"] = df["iceberg_id"].astype(str).str.strip()
    df = df[df["iceberg_id"].str.len() > 0].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df = df[((df["latitude"] >= -90) & (df["latitude"] <= 90)) & ((df["longitude"] >= -180) & (df["longitude"] <= 180))].copy()
    df = df.drop_duplicates(subset=["iceberg_id", "timestamp"]).copy()
    df = df.sort_values(["iceberg_id", "timestamp"]).reset_index(drop=True)

    rows_to_drop = []
    for iceberg_id, group in df.groupby("iceberg_id", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        for idx in range(1, len(group)):
            prev = group.iloc[idx - 1]
            curr = group.iloc[idx]
            dt_hours = (curr["timestamp"] - prev["timestamp"]).total_seconds() / 3600.0
            if pd.isna(dt_hours) or dt_hours <= 0:
                rows_to_drop.append(curr.name)
                continue
            dist_km = haversine_km(float(prev["latitude"]), float(prev["longitude"]), float(curr["latitude"]), float(curr["longitude"]))
            speed_kmh = dist_km / dt_hours
            if speed_kmh > 200.0:
                rows_to_drop.append(curr.name)
    if rows_to_drop:
        df = df.drop(index=sorted(set(rows_to_drop))).copy()

    df = df.sort_values(["iceberg_id", "timestamp"]).reset_index(drop=True)
    LOGGER.info("ICEBERG rows after cleaning: %d", len(df))
    LOGGER.info("ICEBERG unique IDs: %d", df["iceberg_id"].nunique())
    return df


def build_feature_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for iceberg_id, group in df.groupby("iceberg_id", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        for idx in range(1, len(group) - 1):
            prev = group.iloc[idx - 1]
            curr = group.iloc[idx]
            nxt = group.iloc[idx + 1]
            dt_hours = (curr["timestamp"] - prev["timestamp"]).total_seconds() / 3600.0
            lead_hours = (nxt["timestamp"] - curr["timestamp"]).total_seconds() / 3600.0
            if pd.isna(dt_hours) or dt_hours <= 0 or pd.isna(lead_hours) or lead_hours <= 0:
                continue
            prev_lat, prev_lon = float(prev["latitude"]), float(prev["longitude"])
            curr_lat, curr_lon = float(curr["latitude"]), float(curr["longitude"])
            next_lat, next_lon = float(nxt["latitude"]), float(nxt["longitude"])
            drift_km = haversine_km(prev_lat, prev_lon, curr_lat, curr_lon)
            prev_speed_kmh = drift_km / dt_hours
            prev_direction = bearing_deg(prev_lat, prev_lon, curr_lat, curr_lon)
            drift_distance = haversine_km(curr_lat, curr_lon, next_lat, next_lon)
            drift_speed = drift_distance / lead_hours
            drift_direction = bearing_deg(curr_lat, curr_lon, next_lat, next_lon)
            rows.append(
                {
                    "iceberg_id": iceberg_id,
                    "timestamp": curr["timestamp"],
                    "latitude": curr_lat,
                    "longitude": curr_lon,
                    "prev_latitude": prev_lat,
                    "prev_longitude": prev_lon,
                    "time_diff_hours": float(dt_hours),
                    "prev_speed_kmh": float(prev_speed_kmh),
                    "prev_direction_deg": float(prev_direction),
                    "drift_distance_km": float(drift_distance),
                    "drift_speed_kmh": float(drift_speed),
                    "drift_direction_deg": float(drift_direction),
                    "month": curr["timestamp"].month,
                    "season": season_for(curr["timestamp"].month),
                    "target_lat_delta": float(next_lat - curr_lat),
                    "target_lon_delta": float(next_lon - curr_lon),
                    "target_horizon_hours": float(lead_hours),
                    "target_latitude": next_lat,
                    "target_longitude": next_lon,
                }
            )
    feature_df = pd.DataFrame(rows)
    feature_df = feature_df.sort_values(["iceberg_id", "timestamp"]).reset_index(drop=True)
    return feature_df


def chrono_split(feature_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    timestamps = pd.to_datetime(feature_df["timestamp"])
    min_ts = timestamps.min()
    max_ts = timestamps.max()
    span = max_ts - min_ts
    train_end = min_ts + span * 0.70
    val_end = min_ts + span * 0.85
    train = feature_df[timestamps <= train_end].copy()
    val = feature_df[(timestamps > train_end) & (timestamps <= val_end)].copy()
    test = feature_df[timestamps > val_end].copy()
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        n = len(feature_df)
        train = feature_df.iloc[: int(n * 0.7)].copy()
        val = feature_df.iloc[int(n * 0.7): int(n * 0.85)].copy()
        test = feature_df.iloc[int(n * 0.85):].copy()
    LOGGER.info("Chronological split: train=%d val=%d test=%d", len(train), len(val), len(test))
    LOGGER.info("Date range: %s -> %s", min_ts, max_ts)
    return {"train": train, "val": val, "test": test}


def prepare_model_data(split_df: pd.DataFrame, scaler: StandardScaler | None = None):
    features = [
        "prev_latitude",
        "prev_longitude",
        "time_diff_hours",
        "prev_speed_kmh",
        "prev_direction_deg",
        "drift_distance_km",
        "drift_speed_kmh",
        "drift_direction_deg",
        "month",
    ]
    X = split_df[features].copy()
    y = split_df[["target_lat_delta", "target_lon_delta"]].copy()
    if scaler is not None:
        X = pd.DataFrame(scaler.transform(X), columns=features, index=X.index)
    return X, y


def fit_and_eval() -> dict:
    cleaned = load_and_clean()
    feature_df = build_feature_rows(cleaned)
    LOGGER.info("Prepared feature rows: %d", len(feature_df))
    if feature_df.empty:
        raise ValueError("No valid iceberg movement feature rows were produced.")

    splits = chrono_split(feature_df)
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    scaler = StandardScaler()
    X_train_raw, y_train_raw = prepare_model_data(train_df, None)
    scaler.fit(X_train_raw)
    X_train, y_train = prepare_model_data(train_df, scaler)
    X_val, y_val = prepare_model_data(val_df, scaler)
    X_test, y_test = prepare_model_data(test_df, scaler)

    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    val_pred = rf.predict(X_val)
    test_pred = rf.predict(X_test)
    val_lat_pred = val_df["latitude"].to_numpy() + val_pred[:, 0]
    val_lon_pred = val_df["longitude"].to_numpy() + val_pred[:, 1]
    test_lat_pred = test_df["latitude"].to_numpy() + test_pred[:, 0]
    test_lon_pred = test_df["longitude"].to_numpy() + test_pred[:, 1]

    def metrics_for(df: pd.DataFrame, lat_pred: np.ndarray, lon_pred: np.ndarray, true_lat: np.ndarray, true_lon: np.ndarray):
        lat_abs = np.abs(lat_pred - true_lat)
        lon_abs = np.abs(lon_pred - true_lon)
        dist_km = np.array([
            haversine_km(float(lat_pred_i), float(lon_pred_i), float(act_lat), float(act_lon))
            for lat_pred_i, lon_pred_i, act_lat, act_lon in zip(lat_pred, lon_pred, true_lat, true_lon)
        ])
        mae_km = float(np.mean(dist_km))
        rmse_km = float(np.sqrt(np.mean(dist_km ** 2)))
        res = {
            "mae_km": mae_km,
            "rmse_km": rmse_km,
            "lat_mae_deg": float(np.mean(lat_abs)),
            "lon_mae_deg": float(np.mean(lon_abs)),
            "mean_distance_km": float(np.mean(dist_km)),
            "median_distance_km": float(np.median(dist_km)),
            "max_distance_km": float(np.max(dist_km)),
            "mean_distance_nm": float(np.mean(dist_km) / 1.852),
            "samples": int(len(df)),
        }
        # horizon buckets
        horizon_map = {"24h": 24.0, "72h": 72.0, "7d": 168.0}
        for bucket_name, limit in horizon_map.items():
            mask = df["target_horizon_hours"] <= limit
            if mask.any():
                bucket_dist = dist_km[mask.to_numpy()]
                res[f"{bucket_name}_mean_km"] = float(np.mean(bucket_dist))
                res[f"{bucket_name}_median_km"] = float(np.median(bucket_dist))
                res[f"{bucket_name}_samples"] = int(mask.sum())
        return res

    val_true_lat = val_df["target_latitude"].to_numpy()
    val_true_lon = val_df["target_longitude"].to_numpy()
    test_true_lat = test_df["target_latitude"].to_numpy()
    test_true_lon = test_df["target_longitude"].to_numpy()

    baseline_val = np.array([
        haversine_km(float(prev_lat), float(prev_lon), float(act_lat), float(act_lon))
        for prev_lat, prev_lon, act_lat, act_lon in zip(val_df["prev_latitude"], val_df["prev_longitude"], val_true_lat, val_true_lon)
    ])
    baseline_test = np.array([
        haversine_km(float(prev_lat), float(prev_lon), float(act_lat), float(act_lon))
        for prev_lat, prev_lon, act_lat, act_lon in zip(test_df["prev_latitude"], test_df["prev_longitude"], test_true_lat, test_true_lon)
    ])

    val_metrics = metrics_for(val_df, val_lat_pred, val_lon_pred, val_true_lat, val_true_lon)
    test_metrics = metrics_for(test_df, test_lat_pred, test_lon_pred, test_true_lat, test_true_lon)
    val_baseline = float(np.mean(baseline_val))
    test_baseline = float(np.mean(baseline_test))

    report = {
        "dataset": str(REAL_DATA),
        "records_before_cleaning": int(len(cleaned)),
        "records_after_cleaning": int(len(feature_df)),
        "unique_icebergs": int(feature_df["iceberg_id"].nunique()),
        "train_samples": int(len(train_df)),
        "validation_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),
        "split_note": "Equivalent chronological split because the available real iceberg data spans 2022-01 to 2023-12; this project does not contain 2025 observations.",
        "baseline_val_mean_km": val_baseline,
        "baseline_test_mean_km": test_baseline,
        "validation": val_metrics,
        "test": test_metrics,
        "model_beats_baseline": test_metrics["mean_distance_km"] < test_baseline,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "evaluation").mkdir(exist_ok=True, parents=True)
    real_feature_path = PROJECT_ROOT / "data" / "processed" / "features" / "feature_table_real.csv"
    real_feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(real_feature_path, index=False)

    joblib.dump(rf, MODEL_DIR / "rf.joblib")
    joblib.dump({"scalers": scaler, "feature_columns": [
        "prev_latitude",
        "prev_longitude",
        "time_diff_hours",
        "prev_speed_kmh",
        "prev_direction_deg",
        "drift_distance_km",
        "drift_speed_kmh",
        "drift_direction_deg",
        "month",
    ], "seq_len": 5, "horizon": 4}, MODEL_DIR / "scalers.joblib")
    joblib.dump({"scaler": scaler, "feature_columns": [
        "prev_latitude",
        "prev_longitude",
        "time_diff_hours",
        "prev_speed_kmh",
        "prev_direction_deg",
        "drift_distance_km",
        "drift_speed_kmh",
        "drift_direction_deg",
        "month",
    ]}, MODEL_DIR / "preprocessor.joblib")

    run_meta = {
        "model": "random_forest",
        "demo_only": False,
        "tracks_resolved": str(real_feature_path),
        "seq_len": 5,
        "horizon_steps": 4,
        "n_tracks": int(feature_df["iceberg_id"].nunique()),
        "n_sequences": int(len(feature_df)),
        "n_features": 9,
        "metrics": {
            "test": test_metrics,
            "baseline": {"mean_distance_km": test_baseline},
            "comparison": {"model_beats_baseline": report["model_beats_baseline"]},
        },
    }
    with open(MODEL_DIR / "run.json", "w", encoding="utf-8") as fh:
        json.dump(run_meta, fh, indent=2)

    with open(MODEL_DIR / "evaluation" / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump({
            "demo_only": False,
            "model": "random_forest",
            "test": test_metrics,
            "baseline": {"mean_distance_km": test_baseline},
        }, fh, indent=2)

    with open(MODEL_DIR / "comparison_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    with open(MODEL_DIR / "comparison_report.md", "w", encoding="utf-8") as fh:
        fh.write("# Iceberg trajectory model comparison report\n\n")
        fh.write(f"- Records before cleaning: {int(len(cleaned))}\n")
        fh.write(f"- Records after cleaning: {int(len(feature_df))}\n")
        fh.write(f"- Unique iceberg IDs: {int(feature_df['iceberg_id'].nunique())}\n")
        fh.write(f"- Training samples: {int(len(train_df))}\n")
        fh.write(f"- Validation samples: {int(len(val_df))}\n")
        fh.write(f"- Test samples: {int(len(test_df))}\n")
        fh.write(f"- Baseline mean distance (test): {test_baseline:.2f} km\n")
        fh.write(f"- Model mean distance (test): {test_metrics['mean_distance_km']:.2f} km\n")
        fh.write(f"- Model beats baseline: {report['model_beats_baseline']}\n\n")
        fh.write("## Test metrics\n")
        for key, value in test_metrics.items():
            fh.write(f"- {key}: {value}\n")

    LOGGER.info("Real iceberg model saved to %s", MODEL_DIR)
    LOGGER.info("Test mean distance km: %.2f | baseline: %.2f | beats baseline: %s", test_metrics["mean_distance_km"], test_baseline, report["model_beats_baseline"])
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fit_and_eval()
