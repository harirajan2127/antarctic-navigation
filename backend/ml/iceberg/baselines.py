"""Statistical baselines for iceberg trajectory forecasting.

Two baselines are provided so ML skill can only be claimed *relative* to
something simple and honest:

* :class:`PersistenceModel`  — the iceberg stays where it was last seen.
* :class:`CurrentDriftModel` — the iceberg advects at the mean observed ocean
  surface current (uo, vo). This is a zero-tune, physically motivated guess.

Both are applied *after* training (on held-out chronological data only).
"""
from __future__ import annotations

import numpy as np

from .utils import haversine_km
from .trajectory_dataset import IcebergTrack, N_FEATURES, TrajectoryDataset


class RecursiveForecaster:
    """Recursively step an iceberg forward through the dataset feature space.

    At every step the previously predicted point becomes the new tail of the
    window; environmental drivers (uo/vo/u10/v10, dimensions) are carried as
    their last known values because no future forecast fields are available.
    Predictions are made in projected X/Y (km) then inverted to lat/lon.
    """

    def __init__(self, ds: TrajectoryDataset, predictor):
        """
        predictor : callable
            ``predict(seq_batch: np.ndarray [B, T, F], scaled=True) -> np.ndarray [B, 2]``
            returning projected next X/Y in **km** (raw space).
        """
        self.ds = ds
        self.predictor = predictor

    # ------------------------------------------------------------------
    def step_seq(self, tr: IcebergTrack, coords: dict) -> np.ndarray:
        xy = coords["x"]
        n_last = self.ds.seq_len
        n = len(xy)
        rows = []
        for i in range(n - n_last, n):
            if i == 0:
                vx = vy = 0.0
            else:
                dt = float((coords["times"][i] - coords["times"][i - 1]) / np.timedelta64(1, "h"))
                vx = (xy[i] - xy[i - 1]) / max(dt, 1e-9)
                vy = (coords["y"][i] - coords["y"][i - 1]) / max(dt, 1e-9)
            s, c = _sincos(coords["times"][i])
            rows.append([
                xy[i], coords["y"][i], vx, vy,
                coords["uo"], coords["vo"], coords["u10"], coords["v10"],
                coords["length"], coords["width"], s, c,
            ])
        block = np.array(rows, dtype=np.float64)[None, :, :]  # [1, T, F]
        stats = self.ds.scalers
        block = (block - stats["mean"]) / stats["std"]
        return block

    # ------------------------------------------------------------------
    def forecast(
        self,
        tr: IcebergTrack,
        start_obs_idx: int,
        steps: int,
    ) -> list[dict]:
        """Forecast ``steps`` points ahead from the observation at start_obs_idx.

        Returns a list (length ``steps``) of dicts:
        {time, hours_from_current, x_km, y_km, lat, lon, distance_km_from_prev}.
        """
        i = start_obs_idx
        # Only observations up to (and including) start_obs_idx may be used,
        # otherwise true future fixes would leak into the prediction window.
        coords = {
            "times": list(tr.times[: i + 1]),
            "x": list(tr.x[: i + 1]),
            "y": list(tr.y[: i + 1]),
            "uo": float(np.nanmean(tr.uo[i - self.ds.seq_len + 1:i + 1]) if i >= self.ds.seq_len - 1 else tr.uo[i]),
            "vo": float(np.nanmean(tr.vo[i - self.ds.seq_len + 1:i + 1]) if i >= self.ds.seq_len - 1 else tr.vo[i]),
            "u10": float(np.nanmean(tr.u10[max(0, i - self.ds.seq_len + 1):i + 1])),
            "v10": float(np.nanmean(tr.v10[max(0, i - self.ds.seq_len + 1):i + 1])),
            "length": float(tr.length[i]),
            "width": float(tr.width[i]),
        }
        prev_xy = (tr.x[start_obs_idx], tr.y[start_obs_idx])
        out: list[dict] = []
        if np.isfinite(tr.median_dt_hours) and tr.median_dt_hours > 0:
            dt_h = tr.median_dt_hours
        elif start_obs_idx + 1 < len(tr.times):
            dt_h = float((tr.times[start_obs_idx + 1] - tr.times[start_obs_idx]) / np.timedelta64(1, "h"))
        else:
            dt_h = 6.0
        for k in range(1, steps + 1):
            pred = self.predictor(self.step_seq(tr, coords))[0]  # [2] km
            coords["x"].append(float(pred[0]))
            coords["y"].append(float(pred[1]))
            coords["times"].append(coords["times"][-1] + np.timedelta64(int(dt_h * 3600 * 1e6), "us"))
            lat, lon = tr.plane.inverse(pred[0], pred[1])
            lat, lon = float(lat), float(_wrap_lon(lon))
            dist = haversine_km(*tr.plane.inverse(*prev_xy), float(lat), float(lon))
            prev_xy = (pred[0], pred[1])
            out.append({
                "time": str(_ts(coords["times"][-1])),
                "hours_from_current": round(k * dt_h, 3),
                "lat": float(lat),
                "lon": float(lon),
                "x_km": float(pred[0]),
                "y_km": float(pred[1]),
                "distance_km_from_prev": float(dist),
            })
        return out


class PersistenceModel:
    """'Forecast': the iceberg does not move."""

    name = "persistence"

    def __init__(self, ds: TrajectoryDataset | None = None):
        self.ds = ds

    def forecast(self, tr: IcebergTrack, start_obs_idx: int, steps: int) -> list[dict]:
        lat, lon = tr.lat[start_obs_idx], tr.lon[start_obs_idx]
        return [
            {
                "time": "",
                "hours_from_current": 0.0,
                "lat": lat,
                "lon": lon,
                "x_km": tr.x[start_obs_idx],
                "y_km": tr.y[start_obs_idx],
                "distance_km_from_prev": 0.0,
            }
            for _ in range(steps)
        ]


class CurrentDriftModel:
    """Advect the iceberg at the last observed mean ocean current (uo, vo)."""

    name = "current_drift"

    def __init__(self, ds: TrajectoryDataset | None = None):
        self.ds = ds

    def forecast(self, tr: IcebergTrack, start_obs_idx: int, steps: int) -> list[dict]:
        dt_h = tr.median_dt_hours if np.isfinite(tr.median_dt_hours) else 6.0
        lo = max(0, start_obs_idx - self.ds.seq_len + 1) if self.ds else 0
        uo = float(np.nanmean(tr.uo[lo:start_obs_idx + 1]))
        vo = float(np.nanmean(tr.vo[lo:start_obs_idx + 1]))
        # uo/vo are m/s; 1 m/s = 3.6 km/h. Assumes transport at current speed.
        dx_km_per_h = uo * 3.6
        dy_km_per_h = vo * 3.6
        x, y = float(tr.x[start_obs_idx]), float(tr.y[start_obs_idx])
        out = []
        t0 = _ts(tr.times[start_obs_idx])
        for k in range(1, steps + 1):
            x += dx_km_per_h * dt_h
            y += dy_km_per_h * dt_h
            lat, lon = tr.plane.inverse(x, y)
            lon = _wrap_lon(lon)
            out.append({
                "time": "",
                "hours_from_current": round(k * dt_h, 3),
                "lat": float(lat),
                "lon": float(lon),
                "x_km": x,
                "y_km": y,
                "distance_km_from_prev": float(dx_km_per_h * dt_h or dy_km_per_h * dt_h),
            })
        _ = t0
        return out


def _sincos(t):
    from .utils import sin_cos_doy

    s, c = sin_cos_doy(np.array([t], dtype="datetime64[us]"))
    return float(s[0]), float(c[0])


def _wrap_lon(lon):
    from .utils import wrap_lon

    return wrap_lon(lon)


def _ts(t) -> np.datetime64:
    return np.asarray(t, dtype="datetime64[us]")