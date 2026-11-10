"""N-BEATS blocks (Oreshkin et al., 2020) and Cross-Resolution Residual Blocks (CRRB).

The original repository's ``models/nbeats.py`` was an accidental copy of the TCN
file and ``NBeatsBlock`` did not exist, so ``hybrid_model.py`` could not be
imported. This module provides the missing implementation.

* ``NBeatsBlock`` - generic block: FC stack -> (backcast, forecast) with a
  backward residual link (Eq. 4-5).
* ``CrossResolutionBlock`` - CRRB (Sec. 3.5.4, Eq. 21): parallel residual
  paths that see the input at several temporal resolutions (average pooled by
  ``r``); each path predicts a low-resolution backcast/forecast that is
  upsampled back and the paths are summed.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FCStack(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_layers: int, dropout: float):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NBeatsBlock(nn.Module):
    """Generic N-BEATS block with optional conditioning vector (e.g. the RAE)."""

    def __init__(self, backcast_size: int, forecast_size: int, hidden: int = 64, n_layers: int = 4,
                 dropout: float = 0.2, cond_dim: int = 0):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.fc = _FCStack(backcast_size + cond_dim, hidden, n_layers, dropout)
        self.theta_b = nn.Linear(hidden, backcast_size)
        self.theta_f = nn.Linear(hidden, forecast_size)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None):
        h = x if cond is None else torch.cat([x, cond], dim=-1)
        h = self.fc(h)
        return self.theta_b(h), self.theta_f(h)


def _pool(x: torch.Tensor, r: int) -> torch.Tensor:
    """Average-pool a (N, L) tensor by factor r, left-padding with the first value."""
    if r == 1:
        return x
    L = x.shape[-1]
    pad = (-L) % r
    if pad:
        x = torch.cat([x[:, :1].expand(-1, pad), x], dim=-1)
    return x.reshape(x.shape[0], -1, r).mean(-1)


def _upsample(x: torch.Tensor, size: int) -> torch.Tensor:
    if x.shape[-1] == size:
        return x
    return F.interpolate(x.unsqueeze(1), size=size, mode="linear", align_corners=False).squeeze(1)


class CrossResolutionBlock(nn.Module):
    """CRRB: sum of N-BEATS paths operating at resolutions ``resolutions``."""

    def __init__(self, backcast_size: int, forecast_size: int, hidden: int = 64, n_layers: int = 4,
                 dropout: float = 0.2, cond_dim: int = 0, resolutions=(1, 2, 4)):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.resolutions = tuple(int(r) for r in resolutions)
        self.paths = nn.ModuleList(
            NBeatsBlock(math.ceil(backcast_size / r), math.ceil(forecast_size / r), hidden, n_layers, dropout, cond_dim)
            for r in self.resolutions
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None):
        backcast = 0.0
        forecast = 0.0
        for r, path in zip(self.resolutions, self.paths):
            b, f = path(_pool(x, r), cond)
            backcast = backcast + _upsample(b, self.backcast_size)
            forecast = forecast + _upsample(f, self.forecast_size)
        return backcast, forecast
