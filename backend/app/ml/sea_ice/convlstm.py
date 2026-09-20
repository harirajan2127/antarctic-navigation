"""ConvLSTM sea-ice forecaster using PyTorch (skeleton).

Deep-learning model that learns spatio-temporal patterns in sea-ice
concentration fields. The architecture and training loop are defined here;
actual training runs in Phase 3 with a validated dataset.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class ConvLSTM2DCell:
    """A single ConvLSTM2D cell implemented with PyTorch."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for the ConvLSTM model.") from exc

        self.torch = torch
        self.nn = nn
        self.padding = kernel_size // 2
        self.hidden_channels = hidden_channels

        self.W_ii = nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=self.padding)
        self.W_hi = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=self.padding)
        self.b_i = nn.Parameter(torch.zeros(hidden_channels))

        self.W_if = nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=self.padding)
        self.W_hf = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=self.padding)
        self.b_f = nn.Parameter(torch.zeros(hidden_channels))

        self.W_ig = nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=self.padding)
        self.W_hg = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=self.padding)
        self.b_g = nn.Parameter(torch.zeros(hidden_channels))

        self.W_io = nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=self.padding)
        self.W_ho = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=self.padding)
        self.b_o = nn.Parameter(torch.zeros(hidden_channels))

    def forward(self, x, h=None, c=None):
        """Step the ConvLSTM cell."""
        torch = self.torch
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_channels, x.size(2), x.size(3))
        if c is None:
            c = torch.zeros_like(h)

        i = torch.sigmoid(self.W_ii(x) + self.W_hi(h) + self.b_i)
        f = torch.sigmoid(self.W_if(x) + self.W_hf(h) + self.b_f)
        g = torch.tanh(self.W_ig(x) + self.W_hg(h) + self.b_g)
        o = torch.sigmoid(self.W_io(x) + self.W_ho(h) + self.b_o)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTMForecaster:
    """Spatio-temporal sea-ice forecaster built around ConvLSTM2D cells."""

    def __init__(self, sequence_length: int = 10, hidden_channels: int = 32) -> None:
        self.sequence_length = sequence_length
        self.hidden_channels = hidden_channels
        self.cell: Any = None
        self.is_trained = False
        self.metric_note = (
            "Not trained. Metrics will be reported after training on a "
            "validated multi-year sea-ice dataset."
        )

    def build_cell(self, in_channels: int = 1) -> None:
        """Instantiate the ConvLSTM cell (lazy import of torch)."""
        self.cell = ConvLSTM2DCell(in_channels, self.hidden_channels)

    def train(self, samples: list[np.ndarray], labels: list[np.ndarray], epochs: int = 50) -> None:
        """Training loop placeholder. Implemented in Phase 3."""
        raise NotImplementedError(
            "ConvLSTM training requires a curated sequence dataset and is "
            "implemented in Phase 3."
        )

    def predict(self, concentration: list[list[float]], horizon_hours: int) -> list[list[float]]:
        """Predict the concentration field at the horizon.

        Falls back to persistence until training data is available.
        """
        if not self.is_trained:
            return [list(map(float, row)) for row in concentration]
        raise NotImplementedError("Trained ConvLSTM inference is implemented in Phase 3.")

    def skill_note(self) -> str:
        return self.metric_note