"""Multi-resolution enhancements of E-ZERO (thesis Sec. 3.5).

* ``MultiResolutionConv``      - MRC, Eq. 18: parallel dilated causal convolutions, concatenated.
* ``DynamicTemporalPooling``   - DTP, Eq. 19: causal moving-average pooling whose window is a
  function of the input resolution (one window per sample, computed on-device).
* ``ResolutionEmbedding``      - RAE, Eq. 20: learnable embedding of the frequency bucket.
* ``ExogenousMixer``           - Exogenous Mixer block, Eq. 22-26: substitutes known future
  values of exogenous channels, then mixes information across the channels of a group with a
  small TCN and a linear head. Channel mixing is permutation invariant (each channel sees its
  own forecast plus the mean/std of its group), so it works for any number of channels, which
  zero-shot use requires.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ezero.frequency import NUM_BUCKETS, pool_window_table
from ezero.models.tcn import CausalConv1d, TemporalConvNet


class MultiResolutionConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilations=(1, 2, 4, 8)):
        super().__init__()
        n = len(dilations)
        sizes = [out_ch // n] * n
        sizes[0] += out_ch - sum(sizes)
        self.branches = nn.ModuleList(
            CausalConv1d(in_ch, s, kernel_size, d) for s, d in zip(sizes, dilations)
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(torch.cat([b(x) for b in self.branches], dim=1))


class DynamicTemporalPooling(nn.Module):
    """Causal average pooling with a per-sample window ``w = f(resolution)``."""

    def __init__(self, channels: int):
        super().__init__()
        self.register_buffer("windows", torch.tensor(pool_window_table(), dtype=torch.long), persistent=False)
        self.proj = nn.Conv1d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor, freq_bucket: torch.Tensor) -> torch.Tensor:
        # h: (N, C, T); freq_bucket: (N,)
        N, C, T = h.shape
        w = self.windows[freq_bucket.clamp(0, len(self.windows) - 1)]          # (N,)
        cs = torch.cumsum(h, dim=-1)
        cs = torch.cat([torch.zeros_like(cs[..., :1]), cs], dim=-1)           # (N, C, T+1)
        t = torch.arange(1, T + 1, device=h.device)                           # end index (exclusive)
        start = (t.unsqueeze(0) - w.unsqueeze(1)).clamp(min=0)                # (N, T)
        idx_end = t.unsqueeze(0).expand(N, T)
        s_end = torch.gather(cs, 2, idx_end.unsqueeze(1).expand(N, C, T))
        s_start = torch.gather(cs, 2, start.unsqueeze(1).expand(N, C, T))
        count = (idx_end - start).clamp(min=1).unsqueeze(1).to(h.dtype)
        pooled = (s_end - s_start) / count
        return h + self.proj(pooled)


class ResolutionEmbedding(nn.Module):
    def __init__(self, dim: int = 8, num_buckets: int = NUM_BUCKETS):
        super().__init__()
        self.emb = nn.Embedding(num_buckets, dim)
        nn.init.normal_(self.emb.weight, 0.0, 0.1)

    def forward(self, freq_bucket: torch.Tensor) -> torch.Tensor:
        return self.emb(freq_bucket.clamp(0, self.emb.num_embeddings - 1))


def group_stats(x: torch.Tensor, group: torch.Tensor, eps: float = 1e-4):
    """Mean / std of rows of ``x`` (N, H) within each group id (N,), broadcast back to rows."""
    n_groups = int(group.max().item()) + 1 if group.numel() else 0
    ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
    cnt = torch.zeros(n_groups, 1, device=x.device, dtype=x.dtype).index_add_(0, group, ones)
    s1 = torch.zeros(n_groups, x.shape[1], device=x.device, dtype=x.dtype).index_add_(0, group, x)
    s2 = torch.zeros(n_groups, x.shape[1], device=x.device, dtype=x.dtype).index_add_(0, group, x * x)
    mean = s1 / cnt
    var = (s2 / cnt - mean * mean).clamp(min=0.0)
    # eps keeps d(std)/d(var) bounded when all channels of a group forecast (nearly) the same values
    return mean[group], (var + eps).sqrt()[group], cnt[group]


class ExogenousMixer(nn.Module):
    def __init__(self, channels: int = 16, layers: int = 2, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        # inputs: own forecast, group mean, group std, single-channel flag, exogenous flag
        self.tcn = TemporalConvNet(5, [channels] * layers, kernel_size, dropout=dropout)
        self.head = nn.Conv1d(channels, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, forecast: torch.Tensor, group: torch.Tensor,
                exog_mask: torch.Tensor | None = None, exog_future: torch.Tensor | None = None) -> torch.Tensor:
        """forecast: (N, H) normalised forecasts; group: (N,) group ids.

        exog_mask: (N,) bool - channels whose future values are known; exog_future: (N, H)
        normalised true future values for those channels (Eq. 22).
        """
        y = forecast
        if exog_mask is not None and exog_future is not None:
            y = torch.where(exog_mask.unsqueeze(1), exog_future, forecast)
        mean, std, cnt = group_stats(y, group)
        single = (cnt <= 1).to(y.dtype).expand_as(y)
        exog_flag = torch.zeros_like(y) if exog_mask is None else exog_mask.to(y.dtype).unsqueeze(1).expand_as(y)
        z = torch.stack([y, mean, std, single, exog_flag], dim=1)          # (N, 5, H)
        delta = self.head(self.tcn(z)).squeeze(1)
        out = y + delta
        if exog_mask is not None and exog_future is not None:
            out = torch.where(exog_mask.unsqueeze(1), exog_future, out)
        return out
