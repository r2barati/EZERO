"""Full-shot baselines trained on each dataset's train split (thesis Sec. 4.3 "other architectures").

Compact re-implementations of DLinear (Zeng et al., 2023), PatchTST (Nie et al., 2023) and
iTransformer (Liu et al., 2024); TSMixer, TimeMixer, TimesNet, Autoformer and FEDformer live in
``baselines/fullshot_extra.py``. One model is trained per (dataset, horizon) with MSE, Adam, the
learning rate halved after every epoch (Time-Series-Library "type1" schedule) and early stopping
on the validation split (patience 3 epochs, best validation checkpoint kept). To fit Kaggle's session limits training is capped by
``max_steps`` / ``max_epochs``; the caps are recorded in the result rows so the budget is
explicit when comparing with published numbers.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from baselines.base import Forecaster


# --------------------------------------------------------------------------- models
class _DLinear(nn.Module):
    def __init__(self, L, H, kernel=25):
        super().__init__()
        self.kernel = kernel
        self.seasonal = nn.Linear(L, H)
        self.trend = nn.Linear(L, H)

    def forward(self, x):  # x (B, L, C)
        xt = x.transpose(1, 2)                              # (B, C, L)
        pad = (self.kernel - 1) // 2
        front = xt[..., :1].repeat(1, 1, pad)
        end = xt[..., -1:].repeat(1, 1, self.kernel - 1 - pad)
        trend = nn.functional.avg_pool1d(torch.cat([front, xt, end], -1), self.kernel, stride=1)
        seasonal = xt - trend
        return (self.seasonal(seasonal) + self.trend(trend)).transpose(1, 2)


class _RevIN(nn.Module):
    def forward(self, x, mode, stats=None):
        if mode == "norm":
            mu = x.mean(1, keepdim=True)
            sd = x.std(1, keepdim=True, unbiased=False) + 1e-5
            return (x - mu) / sd, (mu, sd)
        mu, sd = stats
        return x * sd + mu


class _PatchTST(nn.Module):
    def __init__(self, L, H, patch=16, stride=8, d_model=128, heads=16, layers=3, ff=256, dropout=0.2):
        super().__init__()
        self.patch, self.stride = patch, stride
        n = (L - patch) // stride + 2
        self.embed = nn.Linear(patch, d_model)
        self.pos = nn.Parameter(torch.randn(1, n, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(n * d_model, H)
        self.drop = nn.Dropout(dropout)
        self.revin = _RevIN()

    def forward(self, x):  # (B, L, C)
        x, st = self.revin(x, "norm")
        B, L, C = x.shape
        z = x.permute(0, 2, 1).reshape(B * C, L)
        z = torch.cat([z, z[:, -1:].expand(-1, self.stride)], dim=1)
        z = z.unfold(1, self.patch, self.stride)            # (B*C, n, patch)
        z = self.encoder(self.drop(self.embed(z) + self.pos))
        y = self.head(z.flatten(1)).reshape(B, C, -1).permute(0, 2, 1)
        return self.revin(y, "denorm", st)


class _ITransformer(nn.Module):
    def __init__(self, L, H, d_model=256, heads=8, layers=2, ff=256, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(L, d_model)
        enc = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, H)
        self.revin = _RevIN()

    def forward(self, x):  # (B, L, C): channels are tokens
        x, st = self.revin(x, "norm")
        z = self.encoder(self.embed(x.transpose(1, 2)))
        return self.revin(self.head(z).transpose(1, 2), "denorm", st)


ARCHS = {"DLinear": _DLinear, "PatchTST": _PatchTST, "iTransformer": _ITransformer}
DEFAULT_LR = {"DLinear": 5e-3, "PatchTST": 1e-4, "iTransformer": 5e-4}

from baselines.fullshot_extra import EXTRA_ARCHS, EXTRA_LR  # noqa: E402

DEFAULT_LR.update(EXTRA_LR)
# models that embed all channels jointly: memory grows with windows, not windows x channels
CHANNEL_MIXING = {"iTransformer", "TSMixer", "TimesNet", "Autoformer", "FEDformer"}


# Dataset-specific sizes stated by the original papers.
# PatchTST (Nie et al., 2023, App. A.1.4): "a reduced size of model (H=4, D=16, F=128)" for ILI, ETTh1, ETTh2.
PAPER_SIZES = {
    ("PatchTST", "ETTh1"): dict(heads=4, d_model=16, ff=128),
    ("PatchTST", "ETTh2"): dict(heads=4, d_model=16, ff=128),
}


def build_arch(arch: str, L: int, H: int, C: int, dataset: str | None = None) -> nn.Module:
    if arch in EXTRA_ARCHS:
        return EXTRA_ARCHS[arch](L, H, C)
    return ARCHS[arch](L, H, **PAPER_SIZES.get((arch, dataset), {}))


# --------------------------------------------------------------------------- forecaster
class FullShotForecaster(Forecaster):
    group = "fullshot"
    zero_shot = False
    multi_horizon = False

    def __init__(self, arch: str, context_length: int = 512, max_epochs: int = 10, max_steps: int = 3000,
                 batch_windows: int = 32, max_series_per_batch: int = 8192, lr: float | None = None,
                 patience: int = 3, device: str | None = None, seed: int = 0, val_max_windows: int = 512):
        self.name = arch
        self.arch = arch
        self.L = context_length
        self.max_epochs, self.max_steps = max_epochs, max_steps
        self.batch_windows = batch_windows
        # PatchTST is channel-independent with 64 patch tokens per series: halve the series per batch
        self.max_series_per_batch = max_series_per_batch if arch != "PatchTST" else min(max_series_per_batch, 4096)
        self.lr = lr or DEFAULT_LR[arch]
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.val_max_windows = val_max_windows
        self.model = None
        self.fit_info = {}

    def torch_module(self):
        return self.model

    def to_device(self, device):
        self.device = device
        if self.model is not None:
            self.model.to(device)
        return self

    def _windows(self, values, L, H):
        n = len(values) - L - H + 1
        return n

    def fit(self, data, horizon: int):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        L, H = self.L, horizon
        train = data.split_values("train")
        val = data.split_values("val")
        C = train.shape[1]
        if self.arch in CHANNEL_MIXING:
            bw = self.batch_windows
        else:
            bw = max(1, min(self.batch_windows, self.max_series_per_batch // C))
        self.model = build_arch(self.arch, L, H, C, data.name).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        tr_t = torch.from_numpy(train).to(self.device)
        va_t = torch.from_numpy(val).to(self.device)
        n_tr = len(train) - L - H + 1
        n_va = len(val) - L - H + 1
        va_idx = np.linspace(0, n_va - 1, num=min(self.val_max_windows, n_va)).astype(int) if n_va > 0 else []
        ar = torch.arange(L + H, device=self.device)

        def batch(t, idx):
            w = t[torch.as_tensor(idx, device=self.device)[:, None] + ar[None, :]]   # (B, L+H, C)
            return w[:, :L], w[:, L:]

        def vloss():
            if len(va_idx) == 0:
                return float("nan")
            self.model.eval()
            tot, n = 0.0, 0
            with torch.no_grad():
                for k in range(0, len(va_idx), bw):
                    x, y = batch(va_t, va_idx[k:k + bw])
                    tot += float(loss_fn(self.model(x), y)) * len(x)
                    n += len(x)
            self.model.train()
            return tot / n

        best, best_state, bad, steps = float("inf"), None, 0, 0
        t0 = time.time()
        epochs = 0
        for ep in range(self.max_epochs):
            epochs = ep + 1
            # Time-Series-Library "type1" schedule: the learning rate halves after every epoch
            for g in opt.param_groups:
                g["lr"] = self.lr * (0.5 ** ep)
            perm = rng.permutation(n_tr)
            for k in range(0, n_tr, bw):
                x, y = batch(tr_t, perm[k:k + bw])
                opt.zero_grad()
                loss = loss_fn(self.model(x), y)
                loss.backward()
                opt.step()
                steps += 1
                if steps >= self.max_steps:
                    break
            vl = vloss()
            if vl < best - 1e-6:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
            if bad >= self.patience or steps >= self.max_steps:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        self.fit_info = dict(train_steps=steps, epochs=epochs, best_val_mse=best, batch_windows=bw,
                             train_seconds=round(time.time() - t0, 1), max_steps=self.max_steps,
                             max_epochs=self.max_epochs, lr=self.lr, lr_schedule="halve_per_epoch")
        return self

    @torch.no_grad()
    def predict(self, context, freq, horizon):
        if self.model is None:  # efficiency benchmark without training
            self.model = build_arch(self.arch, context.shape[1], horizon, context.shape[2]).to(self.device).eval()
        x = torch.from_numpy(context).to(self.device)
        C = context.shape[2]
        bw = self.batch_windows if self.arch in CHANNEL_MIXING else max(1, self.max_series_per_batch // C)
        outs = [self.model(x[k:k + bw]).float().cpu() for k in range(0, len(x), bw)]
        return torch.cat(outs).numpy()[:, :horizon]

    def info(self):
        return dict(self.fit_info)
