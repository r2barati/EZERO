"""Remaining "other architectures" of the thesis (Sec. 4.1 / 4.3): TSMixer, TimeMixer, TimesNet,
Autoformer and FEDformer.

Compact re-implementations that follow the original papers and the structure of the reference code
in the Time-Series-Library (thuml). Calendar (time-stamp) features are not used by any model in this
repository, so the embeddings are value + positional only. Known simplifications, kept identical for
every dataset:

* Autoformer: the batch-shared time-delay aggregation (the reference's training-time variant) is used
  at inference as well.
* FEDformer: Fourier version (FEB-f / FEA-f) with 32 randomly selected modes; single moving-average
  decomposition instead of the mixture of experts.
* Model widths are reduced for a context of 512 (the reference scripts use 96) so that every
  (dataset, horizon) pair trains within the Kaggle budget; widths are listed in ``EXTRA_ARCHS``.

All models take (B, L, C) standard-scaled inputs and return (B, H, C).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Norm(nn.Module):
    """Instance normalisation over time (RevIN without affine / non-stationary normalisation)."""

    def forward(self, x, mode, stats=None):
        if mode == "norm":
            mu = x.mean(1, keepdim=True).detach()
            sd = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5).detach()
            return (x - mu) / sd, (mu, sd)
        mu, sd = stats
        return x * sd + mu


class _MovingAvg(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel

    def forward(self, x):  # (B, T, D)
        pad = (self.kernel - 1) // 2
        front = x[:, :1].repeat(1, pad, 1)
        end = x[:, -1:].repeat(1, self.kernel - 1 - pad, 1)
        return F.avg_pool1d(torch.cat([front, x, end], 1).transpose(1, 2), self.kernel, 1).transpose(1, 2)


class _Decomp(nn.Module):
    def __init__(self, kernel=25):
        super().__init__()
        self.ma = _MovingAvg(kernel)

    def forward(self, x):
        trend = self.ma(x)
        return x - trend, trend


class _PosEmb(nn.Module):
    def __init__(self, d, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)[:, : pe[:, 1::2].shape[1]]
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class _TokenEmb(nn.Module):
    def __init__(self, c_in, d):
        super().__init__()
        self.conv = nn.Conv1d(c_in, d, 3, padding=1, padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class _DataEmb(nn.Module):
    def __init__(self, c_in, d, dropout=0.1, pos=True):
        super().__init__()
        self.tok = _TokenEmb(c_in, d)
        self.pos = _PosEmb(d) if pos else None
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.tok(x)
        if self.pos is not None:
            h = h + self.pos(h)
        return self.drop(h)


# --------------------------------------------------------------------------- TSMixer
class TSMixer(nn.Module):
    """Chen et al. (2023): interleaved time-mixing and feature-mixing MLPs, then a temporal projection."""

    def __init__(self, L, H, C, n_block=2, d_model=256, dropout=0.1):
        super().__init__()
        self.norm = _Norm()
        self.blocks = nn.ModuleList()
        for _ in range(n_block):
            self.blocks.append(nn.ModuleDict({
                "time": nn.Sequential(nn.Linear(L, d_model), nn.ReLU(), nn.Linear(d_model, L), nn.Dropout(dropout)),
                "feat": nn.Sequential(nn.Linear(C, d_model), nn.ReLU(), nn.Linear(d_model, C), nn.Dropout(dropout)),
            }))
        self.proj = nn.Linear(L, H)

    def forward(self, x):
        x, st = self.norm(x, "norm")
        for b in self.blocks:
            x = x + b["time"](x.transpose(1, 2)).transpose(1, 2)
            x = x + b["feat"](x)
        return self.norm(self.proj(x.transpose(1, 2)).transpose(1, 2), "denorm", st)


# --------------------------------------------------------------------------- TimeMixer
class _ScaleMixing(nn.Module):
    """Past-Decomposable-Mixing: bottom-up seasonal mixing, top-down trend mixing (Wang et al., 2024)."""

    def __init__(self, lengths, d_model, d_ff, dropout, kernel=25):
        super().__init__()
        self.decomp = _Decomp(kernel)
        self.lengths = lengths
        self.down = nn.ModuleList(nn.Sequential(nn.Linear(lengths[i], lengths[i + 1]), nn.GELU(),
                                                nn.Linear(lengths[i + 1], lengths[i + 1]))
                                  for i in range(len(lengths) - 1))
        self.up = nn.ModuleList(nn.Sequential(nn.Linear(lengths[i + 1], lengths[i]), nn.GELU(),
                                              nn.Linear(lengths[i], lengths[i]))
                                for i in range(len(lengths) - 1))
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, xs):  # list of (N, T_i, D)
        seas, trend = zip(*[self.decomp(x) for x in xs])
        seas = [s.transpose(1, 2) for s in seas]            # (N, D, T_i)
        trend = [t.transpose(1, 2) for t in trend]
        out_s = [seas[0]]
        hi = seas[0]
        for i in range(len(seas) - 1):
            lo = seas[i + 1] + self.down[i](hi)
            out_s.append(lo)
            hi = lo
        out_t = [None] * len(trend)
        lo = trend[-1]
        out_t[-1] = lo
        for i in reversed(range(len(trend) - 1)):
            hi = trend[i] + self.up[i](lo)
            out_t[i] = hi
            lo = hi
        return [x + self.drop(self.ff((s + t).transpose(1, 2))) for x, s, t in zip(xs, out_s, out_t)]


class TimeMixer(nn.Module):
    """Channel-independent TimeMixer with 3 down-sampling levels (window 2) and Future-Multipredictor-Mixing."""

    def __init__(self, L, H, C, d_model=16, d_ff=32, e_layers=2, levels=3, window=2, dropout=0.1):
        super().__init__()
        self.norm = _Norm()
        self.window, self.levels = window, levels
        lengths = [L // window ** i for i in range(levels + 1)]
        self.lengths = lengths
        self.emb = _DataEmb(1, d_model, dropout, pos=False)
        self.pdm = nn.ModuleList(_ScaleMixing(lengths, d_model, d_ff, dropout) for _ in range(e_layers))
        self.predict = nn.ModuleList(nn.Linear(n, H) for n in lengths)
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x):
        B, L, C = x.shape
        x, st = self.norm(x, "norm")
        z = x.permute(0, 2, 1).reshape(B * C, L, 1)
        xs = [z]
        for _ in range(self.levels):
            z = F.avg_pool1d(z.transpose(1, 2), self.window).transpose(1, 2)
            xs.append(z)
        hs = [self.emb(x_i) for x_i in xs]
        for layer in self.pdm:
            hs = layer(hs)
        y = sum(self.proj(p(h.transpose(1, 2)).transpose(1, 2)) for p, h in zip(self.predict, hs))  # (B*C, H, 1)
        y = y.reshape(B, C, -1).permute(0, 2, 1)
        return self.norm(y, "denorm", st)


# --------------------------------------------------------------------------- TimesNet
class _Inception(nn.Module):
    def __init__(self, c_in, c_out, num_kernels=6):
        super().__init__()
        self.kernels = nn.ModuleList(nn.Conv2d(c_in, c_out, 2 * i + 1, padding=i) for i in range(num_kernels))
        for k in self.kernels:
            nn.init.kaiming_normal_(k.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return torch.stack([k(x) for k in self.kernels], -1).mean(-1)


class _TimesBlock(nn.Module):
    def __init__(self, T, d_model, d_ff, top_k=5, num_kernels=6):
        super().__init__()
        self.T, self.k = T, top_k
        self.conv = nn.Sequential(_Inception(d_model, d_ff, num_kernels), nn.GELU(),
                                  _Inception(d_ff, d_model, num_kernels))

    def forward(self, x):  # (B, T, D)
        B, T, D = x.shape
        xf = torch.fft.rfft(x, dim=1)
        amp = xf.abs().mean(0).mean(-1)
        amp[0] = 0
        top = torch.topk(amp, self.k).indices.detach().cpu().numpy()
        periods = [max(1, T // max(1, int(f))) for f in top]
        weights = xf.abs().mean(-1)[:, top]                      # (B, k)
        res = []
        for p in periods:
            length = T if T % p == 0 else (T // p + 1) * p
            out = torch.cat([x, x.new_zeros(B, length - T, D)], 1) if length != T else x
            out = out.reshape(B, length // p, p, D).permute(0, 3, 1, 2)
            out = self.conv(out).permute(0, 2, 3, 1).reshape(B, -1, D)
            res.append(out[:, :T])
        res = torch.stack(res, -1)
        w = F.softmax(weights, dim=1).unsqueeze(1).unsqueeze(1)
        return (res * w).sum(-1) + x


class TimesNet(nn.Module):
    """Wu et al. (2023): FFT-selected periods, 2-D Inception blocks over (period, phase)."""

    def __init__(self, L, H, C, d_model=16, d_ff=32, e_layers=2, top_k=5, num_kernels=6, dropout=0.1):
        super().__init__()
        self.L, self.H = L, H
        self.norm = _Norm()
        self.emb = _DataEmb(C, d_model, dropout)
        self.extend = nn.Linear(L, L + H)
        self.blocks = nn.ModuleList(_TimesBlock(L + H, d_model, d_ff, top_k, num_kernels) for _ in range(e_layers))
        self.ln = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, C)

    def forward(self, x):
        x, st = self.norm(x, "norm")
        h = self.emb(x)
        h = self.extend(h.transpose(1, 2)).transpose(1, 2)
        for b in self.blocks:
            h = self.ln(b(h))
        y = self.proj(h)[:, -self.H:]
        return self.norm(y, "denorm", st)


# --------------------------------------------------------------------------- Autoformer / FEDformer
class _AutoCorrelation(nn.Module):
    """Period-based dependencies via FFT auto-correlation and time-delay aggregation (Wu et al., 2021)."""

    def __init__(self, factor=1):
        super().__init__()
        self.factor = factor

    def forward(self, q, k, v):  # (B, T, Hh, E)
        B, T, Hh, E = q.shape
        S = k.shape[1]
        if T > S:
            k = torch.cat([k, k.new_zeros(B, T - S, Hh, E)], 1)
            v = torch.cat([v, v.new_zeros(B, T - S, Hh, E)], 1)
        else:
            k, v = k[:, :T], v[:, :T]
        qf = torch.fft.rfft(q.permute(0, 2, 3, 1), dim=-1)
        kf = torch.fft.rfft(k.permute(0, 2, 3, 1), dim=-1)
        corr = torch.fft.irfft(qf * torch.conj(kf), n=T, dim=-1)              # (B, Hh, E, T)
        top_k = max(1, int(self.factor * math.log(T)))
        mean_corr = corr.mean(1).mean(1)                                    # (B, T)
        idx = torch.topk(mean_corr.mean(0), top_k).indices
        w = torch.softmax(mean_corr[:, idx], dim=-1)                       # (B, k)
        vv = v.permute(0, 2, 3, 1)                                          # (B, Hh, E, T)
        agg = torch.zeros_like(vv)
        for i in range(top_k):
            agg = agg + torch.roll(vv, -int(idx[i]), -1) * w[:, i].view(B, 1, 1, 1)
        return agg.permute(0, 3, 1, 2)


class _FourierBlock(nn.Module):
    """FEB-f: learnable complex weights on a random subset of Fourier modes (Zhou et al., 2022)."""

    def __init__(self, d_model, n_heads, T, modes=32, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        m = min(modes, T // 2)
        self.index = sorted(rng.choice(T // 2, m, replace=False).tolist())
        e = d_model // n_heads
        scale = 1 / (d_model * d_model)
        self.w = nn.Parameter(scale * torch.rand(n_heads, e, e, len(self.index), dtype=torch.cfloat))

    def forward(self, q, k, v):  # (B, T, Hh, E) -> mixes q only (self-attention replacement)
        B, T, Hh, E = q.shape
        x = q.permute(0, 2, 3, 1)
        xf = torch.fft.rfft(x, dim=-1)
        out = torch.zeros(B, Hh, E, T // 2 + 1, device=q.device, dtype=torch.cfloat)
        for j, i in enumerate(self.index):
            if i < xf.shape[-1]:
                out[..., j] = torch.einsum("bhi,hio->bho", xf[..., i], self.w[..., j])
        return torch.fft.irfft(out, n=T, dim=-1).permute(0, 3, 1, 2)


class _FourierCross(nn.Module):
    """FEA-f: cross attention in the frequency domain with tanh activation."""

    def __init__(self, d_model, n_heads, Tq, Tkv, modes=32, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.iq = sorted(rng.choice(Tq // 2, min(modes, Tq // 2), replace=False).tolist())
        self.ikv = sorted(rng.choice(Tkv // 2, min(modes, Tkv // 2), replace=False).tolist())
        e = d_model // n_heads
        self.scale = 1 / (d_model * d_model)
        self.w = nn.Parameter(self.scale * torch.rand(n_heads, e, e, len(self.iq), dtype=torch.cfloat))

    def forward(self, q, k, v):
        B, Tq, Hh, E = q.shape
        qf = torch.fft.rfft(q.permute(0, 2, 3, 1), dim=-1)[..., self.iq]
        kf = torch.fft.rfft(k.permute(0, 2, 3, 1), dim=-1)
        kf = kf[..., [i for i in self.ikv if i < kf.shape[-1]]]
        qk = torch.einsum("bhex,bhey->bhxy", qf, kf).tanh()
        qkv = torch.einsum("bhxy,bhey->bhex", qk, kf)
        qkvw = torch.einsum("bhex,heox->bhox", qkv, self.w)
        out = torch.zeros(B, Hh, E, Tq // 2 + 1, device=q.device, dtype=torch.cfloat)
        for j, i in enumerate(self.iq):
            out[..., i] = qkvw[..., j]
        return (torch.fft.irfft(out * self.scale, n=Tq, dim=-1)).permute(0, 3, 1, 2)


class _AttnLayer(nn.Module):
    def __init__(self, inner, d_model, n_heads):
        super().__init__()
        self.inner, self.h = inner, n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        B, T, _ = q.shape
        S = k.shape[1]
        out = self.inner(self.q(q).view(B, T, self.h, -1), self.k(k).view(B, S, self.h, -1),
                         self.v(v).view(B, S, self.h, -1))
        return self.o(out.reshape(B, T, -1))


class _MyLayerNorm(nn.Module):
    """Season-specific layer norm of Autoformer (subtract the time mean after LayerNorm)."""

    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        h = self.ln(x)
        return h - h.mean(1, keepdim=True)


class _EncLayer(nn.Module):
    def __init__(self, attn, d_model, d_ff, kernel, dropout):
        super().__init__()
        self.attn = attn
        self.ff = nn.Sequential(nn.Conv1d(d_model, d_ff, 1, bias=False), nn.GELU(), nn.Dropout(dropout),
                                nn.Conv1d(d_ff, d_model, 1, bias=False))
        self.d1, self.d2 = _Decomp(kernel), _Decomp(kernel)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.d1(x + self.drop(self.attn(x, x, x)))
        x, _ = self.d2(x + self.drop(self.ff(x.transpose(1, 2)).transpose(1, 2)))
        return x


class _DecLayer(nn.Module):
    def __init__(self, self_attn, cross_attn, d_model, c_out, d_ff, kernel, dropout):
        super().__init__()
        self.sa, self.ca = self_attn, cross_attn
        self.ff = nn.Sequential(nn.Conv1d(d_model, d_ff, 1, bias=False), nn.GELU(), nn.Dropout(dropout),
                                nn.Conv1d(d_ff, d_model, 1, bias=False))
        self.d1, self.d2, self.d3 = _Decomp(kernel), _Decomp(kernel), _Decomp(kernel)
        self.drop = nn.Dropout(dropout)
        self.trend_proj = nn.Conv1d(d_model, c_out, 3, padding=1, padding_mode="circular", bias=False)

    def forward(self, x, cross):
        x, t1 = self.d1(x + self.drop(self.sa(x, x, x)))
        x, t2 = self.d2(x + self.drop(self.ca(x, cross, cross)))
        x, t3 = self.d3(x + self.drop(self.ff(x.transpose(1, 2)).transpose(1, 2)))
        trend = self.trend_proj((t1 + t2 + t3).transpose(1, 2)).transpose(1, 2)
        return x, trend


class _DecompTransformer(nn.Module):
    """Shared encoder-decoder of Autoformer and FEDformer (seasonal-trend decomposition architecture)."""

    def __init__(self, L, H, C, kind, d_model=128, n_heads=8, e_layers=2, d_layers=1, d_ff=256, kernel=25,
                 dropout=0.1, modes=32):
        super().__init__()
        self.L, self.H = L, H
        self.label = L // 2
        Td = self.label + H
        self.decomp = _Decomp(kernel)
        self.enc_emb = _DataEmb(C, d_model, dropout, pos=False)
        self.dec_emb = _DataEmb(C, d_model, dropout, pos=False)

        def enc_inner(i):
            return _AutoCorrelation() if kind == "auto" else _FourierBlock(d_model, n_heads, L, modes, seed=i)

        def dec_self(i):
            return _AutoCorrelation() if kind == "auto" else _FourierBlock(d_model, n_heads, Td, modes, seed=100 + i)

        def dec_cross(i):
            return _AutoCorrelation() if kind == "auto" else _FourierCross(d_model, n_heads, Td, L, modes, seed=200 + i)

        self.enc = nn.ModuleList(_EncLayer(_AttnLayer(enc_inner(i), d_model, n_heads), d_model, d_ff, kernel, dropout)
                                 for i in range(e_layers))
        self.enc_norm = _MyLayerNorm(d_model)
        self.dec = nn.ModuleList(_DecLayer(_AttnLayer(dec_self(i), d_model, n_heads),
                                           _AttnLayer(dec_cross(i), d_model, n_heads), d_model, C, d_ff, kernel,
                                           dropout) for i in range(d_layers))
        self.dec_norm = _MyLayerNorm(d_model)
        self.out = nn.Linear(d_model, C)
        self.norm = _Norm()

    def forward(self, x):
        x, st = self.norm(x, "norm")
        B, L, C = x.shape
        mean = x.mean(1, keepdim=True).repeat(1, self.H, 1)
        seas, trend = self.decomp(x)
        trend_init = torch.cat([trend[:, -self.label:], mean], 1)
        seas_init = torch.cat([seas[:, -self.label:], x.new_zeros(B, self.H, C)], 1)
        e = self.enc_emb(x)
        for layer in self.enc:
            e = layer(e)
        e = self.enc_norm(e)
        d = self.dec_emb(seas_init)
        trend_acc = trend_init
        for layer in self.dec:
            d, t = layer(d, e)
            trend_acc = trend_acc + t
        y = (self.out(self.dec_norm(d)) + trend_acc)[:, -self.H:]
        return self.norm(y, "denorm", st)


def Autoformer(L, H, C, **kw):
    return _DecompTransformer(L, H, C, "auto", **kw)


def FEDformer(L, H, C, **kw):
    return _DecompTransformer(L, H, C, "fourier", **kw)


EXTRA_ARCHS = {
    "TSMixer": TSMixer,         # n_block 2, d_model 256
    "TimeMixer": TimeMixer,     # d_model 16, d_ff 32, 3 down-sampling levels
    "TimesNet": TimesNet,       # d_model 16, d_ff 32, top-k 5, 6 inception kernels
    "Autoformer": Autoformer,   # d_model 128, 8 heads, 2+1 layers, d_ff 256
    "FEDformer": FEDformer,     # d_model 128, 8 heads, 2+1 layers, 32 Fourier modes
}
EXTRA_LR = {"TSMixer": 1e-3, "TimeMixer": 1e-2, "TimesNet": 1e-3, "Autoformer": 1e-4, "FEDformer": 1e-4}
