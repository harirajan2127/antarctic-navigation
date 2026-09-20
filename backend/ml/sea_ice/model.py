"""PyTorch models for sea-ice forecasting.

Implements the small CPU-friendly ConvLSTM (Shi et al., 2015) used as the
primary forecasting model and a trivial persistence head used as Model A.

Tensor layouts follow the task specification:

* ConvLSTM input  : ``[batch, time, height, width, channels]``
* ConvLSTM output : ``[batch, height, width, 1]``
* Persistence     : returns the last observed concentration (``[batch, H, W, 1]``)
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

import torch
import torch.nn as nn

LOG = logging.getLogger(__name__)


class ConvLSTMCell(nn.Module):
    """Classic ConvLSTM cell: 3x3 convs on the concatenated [h, x] state."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,  # i, f, g, o gates
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, g, o = gates.chunk(4, dim=1)
        i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

    def initial_state(
        self, batch: int, height: int, width: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (batch, self.hidden, height, width)
        return (torch.zeros(shape, device=device), torch.zeros(shape, device=device))


class ConvLSTM(nn.Module):
    """Single-layer ConvLSTM encoder + 1x1 convolution head.

    Parameters
    ----------
    in_channels : number of feature channels per (time, cell), i.e. F
    hidden_channels : width of the spatio-temporal hidden state
    kernel_size : convolution kernel (default 3 -> 3x3)
    sigmoid_output : bound predictions to [0, 1] (sea-ice concentration)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 16,
        kernel_size: int = 3,
        sigmoid_output: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.sigmoid_output = sigmoid_output
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size)
        self.head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, H, W, F] -> [B, H, W, 1]."""
        x = x.permute(0, 4, 1, 2, 3)  # [B, T, H, W, F] -> [B, F, T, H, W]
        b, _, t, h, w = x.shape
        h_state, c_state = self.cell.initial_state(b, h, w, x.device)
        for s in range(t):
            h_state, c_state = self.cell(x[:, :, s], (h_state, c_state))
        out = self.head(h_state)  # [B, 1, H, W]
        return out.permute(0, 2, 3, 1)  # [B, H, W, 1]

    def predict_grid(
        self, x: torch.Tensor, clip01: bool = True
    ) -> torch.Tensor:
        """Inference helper: [1, T, H, W, F] -> [1, H, W, 1] with sigmoid."""
        self.eval()
        with torch.no_grad():
            out = torch.sigmoid(self.forward(x)) if self.sigmoid_output else self.forward(x)
        if clip01:
            out = out.clamp(0.0, 1.0)
        return out


class PersistenceModel(nn.Module):
    """Model A: 'tomorrow equals today' - returns the last observed grid."""

    def __init__(self, target_channel: int = 0):
        super().__init__()
        self.target_channel = target_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, H, W, F] -> [B, H, W, 1] using x[:, -1, :, :, 0]."""
        last = x[:, -1, :, :, self.target_channel]
        return last.unsqueeze(-1)

    def predict_grid(self, x: torch.Tensor, clip01: bool = True) -> torch.Tensor:
        with torch.no_grad():
            out = self.forward(x)
        return out.clamp(0.0, 1.0) if clip01 else out


def build_model(name: str, config: Mapping[str, Any], device: torch.device) -> nn.Module:
    """Instantiate a model given a config dict with at least ``in_channels``."""
    cfg = {
        "in_channels": int(config.get("in_channels", 11)),
        "hidden_channels": int(config.get("hidden_channels", 16)),
        "kernel_size": int(config.get("kernel_size", 3)),
        "sigmoid_output": bool(config.get("sigmoid_output", True)),
    }
    name = name.lower()
    if name == "convlstm":
        model = ConvLSTM(**cfg)
    elif name == "persistence":
        model = PersistenceModel()
    else:
        raise ValueError(f"Unknown model '{name}' (choose convlstm | persistence | random_forest)")
    model.to(device)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)