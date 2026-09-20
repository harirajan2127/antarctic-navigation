# Model Training

Two forecasting pipelines live in `backend/ml/` (`sea_ice/` and `iceberg/`).
All commands run as modules from `backend/` and write artifacts under
`backend/models/`.

> Honesty rule: with the shipped synthetic data every training run requires
> `--demo` and produces **code-path-check numbers only** — no real skill is
> claimed. Real training only needs real processed inputs in
> `backend/datasets/processed/` and then runs without `--demo`.

## Sea-ice concentration

Consumes `backend/datasets/processed/sea_ice.nc`.

| Model | Type | Input |
|---|---|---|
| A | Persistence (`tomorrow = today`) | last observed SIC |
| B | Random Forest | per-cell tabular features (SIC history/change, ocean T/currents, wind, time) |
| C | ConvLSTM (PyTorch, CPU-friendly) | `[batch, time, height, width, features]` → `[batch, height, width, 1]` |

Training: strictly chronological train/val/test split, per-channel scalers fit
on train only (saved with the model), masked loss for invalid grid cells,
early stopping, configurable batch size, CPU/GPU, reproducibility seed.

```bash
cd backend
python -m ml.sea_ice.train --out models/sea_ice/run --demo --model convlstm --with-rf
python -m ml.sea_ice.train --out models/sea_ice/run --demo --model random_forest
python -m ml.sea_ice.evaluate --out models/sea_ice/run
python -m ml.sea_ice.predict --out models/sea_ice/run
python -m ml.sea_ice.predict --out models/sea_ice/run --date 2026-09-11
```

Outputs: checkpoint + scalers under `run/`, evaluation (MAE/RMSE/spatial
correlation, error maps, loss curves) under `run/evaluation/`, prediction
NetCDF + JSON under `run/predictions/`.

Runtime: `SeaIceRuntime.grid_predict()` serves the model's native **24 h**
update step through `/api/sea-ice/predict` and `/api/sea-ice/forecast`; if the
checkpoint is missing or malformed the runtime reports `available:false` and
callers fall back to persistence.

## Iceberg trajectory

Consumes `backend/datasets/processed/features/feature_table.csv` (tracks + ocean
current + weather drivers projected into a local tangent plane in km).

| Model | Notes |
|---|---|
| Persistence | baseline — position unchanged |
| Current-drift | baseline — advects at observed ocean current |
| Random Forest | two independent forests (X, Y) |
| LSTM (PyTorch) | stacked LSTM + FC head, recursive multi-step |

Features per timestep: `[x, y, vx, vy, uo, vo, u10, v10, length, width, sin_doy, cos_doy]`.

```bash
cd backend
python -m ml.iceberg.train_rf   --demo --out models/iceberg/run
python -m ml.iceberg.train_lstm --demo --out models/iceberg/run --epochs 60
python -m ml.iceberg.evaluate   --out models/iceberg/run
python -m ml.iceberg.predict --iceberg-id DEMO-B000 --steps 4 --model random_forest
python -m ml.iceberg.predict --iceberg-id DEMO-B000 --steps 8 --model lstm
```

Outputs: `run/evaluation/{metrics.json, metrics.md, error_by_horizon.png}`,
predictions per iceberg. Runtime (`IcebergRuntime`) does recursive multi-step
forecasting at a **6 h** reference step and serves
`/api/icebergs/predict` when the iceberg exists in the feature table.

## Model registry & honesty

- `/api/models/status` lists the sea-ice and iceberg checkpoints actually on
  disk with `trained_on_demo`, `available`, and active config.
- `/api/analytics/model-metrics` returns seeded validation metrics; nothing is
  fabricated, and the frontend Analytics page renders only what the registry
  actually reports.
- Current shipped artifacts are demo-trained (`trained_on_demo: true`) and the
  UI labels them as such.