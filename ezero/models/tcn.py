"""Causal Temporal Convolutional Network (Bai et al., 2018).

Rewritten from the original ``models/tcn.py``: left padding replaces the
pad-then-chomp trick (same causal behaviour, no wasted computation), dilations
are taken from the config instead of being hard-coded, and the deprecated
``torch.nn.utils.weight_norm`` is replaced by the parametrization API.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # torch >= 2.1
    from torch.nn.utils.parametrizations import weight_norm as _weight_norm
except ImportError:  # pragma: no cover - old torch
    from torch.nn.utils import weight_norm as _weight_norm


class CausalConv1d(nn.Module):
    """1-D convolution whose output at time t only depends on inputs <= t (Eq. 6-7)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1, weight_norm: bool = True):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        nn.init.normal_(conv.weight, 0.0, 0.01)
        nn.init.zeros_(conv.bias)
        self.conv = _weight_norm(conv) if weight_norm else conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_pad, 0)))


class TemporalBlock(nn.Module):
    """Two dilated causal convolutions with ReLU/dropout and a residual connection (Eq. 8)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_ch, out_ch, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(out_ch, out_ch, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0.0, 0.01)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(self.net(x) + res)


class TemporalConvNet(nn.Module):
    """Stack of TemporalBlocks with (by default exponentially increasing) dilations."""

    def __init__(self, in_ch: int, channels: list[int], kernel_size: int = 3,
                 dilations: list[int] | None = None, dropout: float = 0.2):
        super().__init__()
        if dilations is None:
            dilations = [2 ** i for i in range(len(channels))]
        if len(dilations) != len(channels):
            raise ValueError("dilations and channels must have the same length")
        layers = []
        prev = in_ch
        for ch, d in zip(channels, dilations):
            layers.append(TemporalBlock(prev, ch, kernel_size, d, dropout))
            prev = ch
        self.network = nn.Sequential(*layers)
        self.out_channels = prev
        self.receptive_field = 1 + 2 * (kernel_size - 1) * sum(dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
