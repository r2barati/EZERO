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
    """Two dilated causal convolutions with ReLU/dropout and a residual connection (Eq. 8).

    With ``cond_dim > 0`` the block implements the conditional convolutions of thesis Sec. 3.3.4
    ("the filters adjust based on the covariates"): a per-output-channel scale and shift are
    generated from the conditioning vector. Scaling output channel c by g_c is the same as
    scaling filter c of the convolution by g_c, so this adjusts the filters per input series
    without materialising a separate kernel for every sample. It starts as the identity
    (g = 1, b = 0), so adding it does not change an already trained block.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float = 0.2,
                 cond_dim: int = 0):
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
        self.film = nn.Linear(cond_dim, 2 * out_ch) if cond_dim else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        res = x if self.downsample is None else self.downsample(x)
        h = self.net(x)
        if self.film is not None:
            if cond is None:
                raise ValueError("conditional convolutions are enabled but no conditioning vector was given")
            gain, bias = self.film(cond).chunk(2, dim=-1)                 # (N, C) each
            h = h * (1.0 + gain).unsqueeze(-1) + bias.unsqueeze(-1)
        return self.relu(h + res)


class TemporalConvNet(nn.Module):
    """Stack of TemporalBlocks with (by default exponentially increasing) dilations."""

    def __init__(self, in_ch: int, channels: list[int], kernel_size: int = 3,
                 dilations: list[int] | None = None, dropout: float = 0.2, cond_dim: int = 0):
        super().__init__()
        if dilations is None:
            dilations = [2 ** i for i in range(len(channels))]
        if len(dilations) != len(channels):
            raise ValueError("dilations and channels must have the same length")
        layers = []
        prev = in_ch
        for ch, d in zip(channels, dilations):
            layers.append(TemporalBlock(prev, ch, kernel_size, d, dropout, cond_dim))
            prev = ch
        # ModuleList, not Sequential: the blocks take the conditioning vector as a second argument.
        # The parameter names are the same either way ("network.0.net.0...."), so checkpoints load.
        self.network = nn.ModuleList(layers)
        self.out_channels = prev
        self.receptive_field = 1 + 2 * (kernel_size - 1) * sum(dilations)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.network:
            x = block(x, cond)
        return x
