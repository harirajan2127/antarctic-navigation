"""PyTorch LSTM for iceberg next-position (and multi-step) forecasting.

Architecture
------------
[x, y, vx, vy, uo, vo, u10, v10, length, width, sin_doy, cos_doy] (per step)
    -> LSTM(hidden_size, num_layers, dropout)
    -> last hidden state
    -> Linear(hidden_size, 2)  -> projected next (X, Y) in km

Training notes
--------------
* sequences are standardised with the dataset scalers (train-only statistics);
* targets (X, Y) stay in raw km so the MSE loss is meaningful in physical units;
* one-step-ahead training; multi-step trajectories are produced by recursive
  forecasting (see :class:`RecursiveForecaster`).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .trajectory_dataset import N_FEATURES, TrajectoryDataset

LOG = logging.getLogger(__name__)


class LSTMPositionModel(nn.Module):
    def __init__(
        self,
        input_size: int = N_FEATURES,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, T, F] -> [B, 2] projected km."""
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


def train_loop(
    ds: TrajectoryDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.0,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 8,
    min_delta: float = 1e-5,
    device: str = "auto",
    seed: int = 7,
    out_dir: Path | None = None,
):
    """Train with early stopping; checkpoints the best state_dict. Returns dict."""
    from .utils import set_seed

    set_seed(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    LOG.info("LSTM training on device=%s", device)

    model = LSTMPositionModel(
        input_size=ds.seq.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xtr, Ytr = ds.seq_tensor(train_idx)
    Xva, Yva = ds.seq_tensor(val_idx)
    Xtr, Ytr, Xva, Yva = (Xtr.to(device), Ytr.to(device), Xva.to(device), Yva.to(device))
    ymean = float(Ytr.mean())
    ystd = float(max(Ytr.std(), 1e-8))

    best_val, best_state, patience_left, history = float("inf"), None, patience, []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        epoch_loss = 0.0
        n_batches = 0
        for s in range(0, len(Xtr), batch_size):
            idx = perm[s:s + batch_size]
            opt.zero_grad()
            pred = model(Xtr[idx])
            loss = loss_fn(pred, Ytr[idx])
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach()) * len(idx)
            n_batches += len(idx)
        train_rmse = float(np.sqrt(epoch_loss / max(n_batches, 1)))
        val_rmse = float(np.sqrt(loss_fn(model(Xva), Yva).item()))

        # mean-absolute error in km and in % of scale
        with torch.no_grad():
            mae_km = float((model(Xva) - Yva).abs().mean().item())

        history.append({
            "epoch": epoch,
            "train_rmse_km": round(train_rmse, 6),
            "val_rmse_km": round(val_rmse, 6),
            "val_mae_km": round(mae_km, 6),
            "val_mae_norm": round(mae_km / ystd, 6),
        })
        improved = val_rmse < best_val - min_delta
        if improved:
            best_val, best_state, patience_left = val_rmse, {k: v.cpu().clone() for k, v in model.state_dict().items()}, patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                LOG.info("Early stop at epoch %d (best val RMSE %.4f km)", epoch, best_val)
                break
        if out_dir is not None:
            torch.save(model.state_dict(), Path(out_dir) / "lstm_last.pt")

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if out_dir is not None:
        torch.save(best_state, Path(out_dir) / "lstm.pt")

    return {
        "history": history,
        "model": model,
        "best_val_rmse_km": best_val,
        "best_epoch": max(h["epoch"] for h in history),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "device": device,
        "ymean": ymean,
        "ystd": ystd,
    }


def predict_batch_km(model: nn.Module, X: np.ndarray | torch.Tensor, device: str = "auto") -> np.ndarray:
    """Forward helper: scaled [B, T, F] -> raw [B, 2] km."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    with torch.no_grad():
        t = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
        y = model(t).cpu().numpy()
    return np.asarray(y)