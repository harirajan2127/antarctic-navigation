"""LSTM iceberg trajectory predictor using PyTorch (skeleton)."""
from __future__ import annotations

from typing import Any

import numpy as np


class LSTMTrajectoryNet:
    """A small stacked LSTM that maps a position history to a displacement."""

    def __init__(self, input_size: int = 4, hidden_size: int = 64, num_layers: int = 2):
        try:
            import torch.nn as nn
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for the LSTM iceberg model.") from exc

        self.nn = nn
        self.net = nn.Sequential(
            nn.LSTM(input_size, hidden_size, num_layers, batch_first=True),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x):
        """Map a (batch, seq, features) tensor to (batch, 2) displacement."""
        lstm_out, _ = self.net[0](x)
        last = lstm_out[:, -1, :]
        return self.net[1](last)


class LSTMIcebergPredictor:
    """Learns iceberg dynamics from observed position histories."""

    def __init__(self, seq_len: int = 10) -> None:
        self.seq_len = seq_len
        self.model: Any = None
        self.is_trained = False
        self.metric_note = (
            "Not trained. Metrics will be reported after training on "
            "validated iceberg track data."
        )

    def train(
        self,
        sequences: list[np.ndarray],
        targets: list[np.ndarray],
        epochs: int = 100,
        lr: float = 1e-3,
    ) -> None:
        """Training loop placeholder; implemented in Phase 3."""
        raise NotImplementedError(
            "LSTM training requires curated iceberg track histories and is "
            "implemented in Phase 3."
        )

    def predict_position(self, iceberg: dict, horizon_hours: int):
        """Predict position. Falls back to persistence when untrained."""
        if not self.is_trained:
            lat = iceberg.get("latitude", 0.0)
            lon = iceberg.get("longitude", 0.0)
            speed = iceberg.get("drift_speed_km_per_hour", 1.0)
            direction = iceberg.get("drift_direction_deg", 225.0)
            from app.ml.iceberg.persistence import PersistenceIcebergPredictor

            return PersistenceIcebergPredictor()._destination_point(
                lat, lon, direction, speed * horizon_hours
            )

        raise NotImplementedError("Trained LSTM inference is implemented in Phase 3.")

    def confidence_note(self) -> str:
        return self.metric_note