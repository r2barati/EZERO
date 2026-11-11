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
* ``FeatureAttention``         - Sec. 3.3.3: "an attention mechanism to dynamically weigh the
  importance of different features". Scores every channel of a group from its *context* and
  replaces the unweighted group mean by an attention-weighted one.
* ``SimilarityMatching``       - Sec. 3.3.5: a bank of prototype embeddings learned during
  pre-training; a new series is matched to the closest ones and the resulting embedding
  conditions the forecast ("adaptive inference").
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def grouped_softmax(score: torch.Tensor, group: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Softmax of ``score`` (N,) within each group id, returned per row (N,)."""
    m = score.new_full((n_groups,), float("-inf"))
    m = m.scatter_reduce(0, group, score.detach(), reduce="amax", include_self=False)
    e = torch.exp(score - m[group])
    denom = torch.zeros(n_groups, device=score.device, dtype=score.dtype).index_add_(0, group, e)
    return e / denom[group].clamp(min=1e-12)


class FeatureAttention(nn.Module):
    """Thesis Sec. 3.3.3: weigh the importance of the different features (channels) of a group.

    Each channel is scored from a summary of its own *context* window, the scores are softmaxed
    over the channels of the group, and the group's forecasts are averaged with those weights.
    Scoring from the context (not from the forecast) keeps the block causal in time, so a shorter
    horizon is still exactly the prefix of a longer one.

    Permutation invariant and independent of the number of channels, as zero-shot use requires:
    the same block handles the 7 channels of ETT and the 862 of Traffic.
    """

    def __init__(self, feat_dim: int, hidden: int = 16):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(feat_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)   # starts as the unweighted mean

    def forward(self, y: torch.Tensor, feat: torch.Tensor, group: torch.Tensor, n_groups: int):
        """y: (N, H) forecasts; feat: (N, D) per-channel context summary; group: (N,)."""
        alpha = grouped_softmax(self.score(feat).squeeze(-1), group, n_groups)      # (N,)
        weighted = torch.zeros(n_groups, y.shape[1], device=y.device, dtype=y.dtype)
        weighted = weighted.index_add(0, group, y * alpha.unsqueeze(1))             # (G, H)
        return weighted[group], alpha


class SimilarityMatching(nn.Module):
    """Thesis Sec. 3.3.5: match a new series to the embeddings learned during pre-training.

    A bank of ``n_prototypes`` embeddings is learned with the model. At inference the context
    window is encoded, matched to the prototypes by cosine similarity, and the softmax-weighted
    prototype is returned as a conditioning vector. This is "adaptive inference": the vector is
    recomputed for every input series, from the context only, so nothing about the forecast
    horizon is used and the model stays zero-shot (no gradient step at test time).
    """

    def __init__(self, dim: int = 16, n_prototypes: int = 32, pool: int = 16, hidden: int = 64):
        super().__init__()
        self.pool = pool
        self.enc = nn.Sequential(nn.Linear(2 * pool, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, dim) * 0.1)
        self.log_temperature = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        """x: (N, L) normalised context (zero where unobserved); observed: (N, L) -> (N, dim)."""
        v = F.adaptive_avg_pool1d(x.unsqueeze(1), self.pool).squeeze(1)
        o = F.adaptive_avg_pool1d(observed.to(x.dtype).unsqueeze(1), self.pool).squeeze(1)
        q = F.normalize(self.enc(torch.cat([v, o], dim=-1)), dim=-1)
        p = F.normalize(self.prototypes, dim=-1)
        alpha = torch.softmax(q @ p.t() / self.log_temperature.exp().clamp(min=1e-2), dim=-1)
        return alpha @ self.prototypes


class ExogenousMixer(nn.Module):
    def __init__(self, channels: int = 16, layers: int = 2, kernel_size: int = 3, dropout: float = 0.1,
                 feat_dim: int = 0):
        super().__init__()
        # inputs: own forecast, group mean, group std, single-channel flag, exogenous flag
        # (+ the attention-weighted group mean when feature attention is enabled)
        self.attn = FeatureAttention(feat_dim) if feat_dim else None
        self.tcn = TemporalConvNet(5 + (1 if feat_dim else 0), [channels] * layers, kernel_size, dropout=dropout)
        self.head = nn.Conv1d(channels, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, forecast: torch.Tensor, group: torch.Tensor,
                exog_mask: torch.Tensor | None = None, exog_future: torch.Tensor | None = None,
                feat: torch.Tensor | None = None) -> torch.Tensor:
        """forecast: (N, H) normalised forecasts; group: (N,) group ids.

        exog_mask: (N,) bool - channels whose future values are known; exog_future: (N, H)
        normalised true future values for those channels (Eq. 22). feat: (N, D) per-channel
        context summary used by the feature-attention block (Sec. 3.3.3).
        """
        y = forecast
        if exog_mask is not None and exog_future is not None:
            y = torch.where(exog_mask.unsqueeze(1), exog_future, forecast)
        mean, std, cnt = group_stats(y, group)
        single = (cnt <= 1).to(y.dtype).expand_as(y)
        exog_flag = torch.zeros_like(y) if exog_mask is None else exog_mask.to(y.dtype).unsqueeze(1).expand_as(y)
        chans = [y, mean, std, single, exog_flag]
        if self.attn is not None:
            if feat is None:
                raise ValueError("feature attention is enabled but no per-channel features were given")
            n_groups = int(group.max().item()) + 1 if group.numel() else 0
            attn_mean, _ = self.attn(y, feat, group, n_groups)
            chans.append(attn_mean)
        z = torch.stack(chans, dim=1)                                      # (N, 5 or 6, H)
        delta = self.head(self.tcn(z)).squeeze(1)
        out = y + delta
        if exog_mask is not None and exog_future is not None:
            out = torch.where(exog_mask.unsqueeze(1), exog_future, out)
        return out
