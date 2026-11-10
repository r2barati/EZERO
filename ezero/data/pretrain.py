"""Fast window sampler for pre-training on a list of panels.

A batch is a set of *groups*; each group is a window cut at the same time step from
up to ``max_group_channels`` channels of one panel, so the Exogenous Mixer learns
cross-channel structure. Datasets are sampled with temperature ``alpha`` on their
number of observations (``alpha=0`` -> uniform over datasets, ``1`` -> proportional),
which balances frequencies the way the thesis' "diverse resolution" sampling intends.

The last ``val_fraction`` of every panel is reserved for validation targets
(context may reach back into the training part, targets never do), so early
stopping uses future, unseen values.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ezero.data.lotsa import Panel


class WindowSampler:
    def __init__(self, panels: list[Panel], context_length: int = 512, horizon: int = 720,
                 min_context: int = 32, min_target: int = 1, max_group_channels: int = 8,
                 alpha: float = 0.5, val_fraction: float = 0.1, seed: int = 0):
        if not panels:
            raise ValueError("no panels to sample from")
        self.panels = panels
        self.L, self.H = context_length, horizon
        self.min_context, self.min_target = min_context, min_target
        self.G = max_group_channels
        self.rng = np.random.default_rng(seed)
        self.split = [max(min_context + min_target, int(p.length * (1.0 - val_fraction))) for p in panels]
        by_ds: dict[str, list[int]] = defaultdict(list)
        for i, p in enumerate(panels):
            by_ds[p.dataset].append(i)
        self.datasets = sorted(by_ds)
        sizes = np.array([sum(panels[i].length * panels[i].channels for i in by_ds[d]) for d in self.datasets], float)
        w = sizes ** alpha
        self.ds_prob = w / w.sum()
        self.ds_panels = [np.array(by_ds[d]) for d in self.datasets]
        self.ds_panel_prob = []
        for idx in self.ds_panels:
            pw = np.array([panels[i].length * panels[i].channels for i in idx], float)
            self.ds_panel_prob.append(pw / pw.sum())

    def describe(self) -> dict:
        return {d: round(float(p), 4) for d, p in zip(self.datasets, self.ds_prob)}

    def _cut_range(self, i: int, split: str):
        T = self.panels[i].length
        s = self.split[i]
        if split == "train":
            return self.min_context, s - self.min_target
        return max(self.min_context, s), T - self.min_target

    def sample(self, batch_size: int, split: str = "train"):
        """Return dict of numpy arrays: context (N,L), target (N,H), freq (N,), group (N,)."""
        ctxs, tgts, freqs, groups = [], [], [], []
        n, g = 0, 0
        tries = 0
        while n < batch_size and tries < batch_size * 50:
            tries += 1
            d = self.rng.choice(len(self.datasets), p=self.ds_prob)
            i = int(self.rng.choice(self.ds_panels[d], p=self.ds_panel_prob[d]))
            lo, hi = self._cut_range(i, split)
            if hi <= lo:
                continue
            t = int(self.rng.integers(lo, hi + 1))
            p = self.panels[i]
            limit = self.split[i] if split == "train" else p.length
            k = min(self.G, p.channels, batch_size - n)
            ch = self.rng.choice(p.channels, size=k, replace=False)
            c0 = max(0, t - self.L)
            ctx = p.values[ch, c0:t]
            tgt = p.values[ch, t:min(t + self.H, limit)]
            with np.errstate(all="ignore"):
                spread = np.nanmax(ctx, axis=1) - np.nanmin(ctx, axis=1) if ctx.shape[1] else np.zeros(len(ch))
            # drop channels with too few observations or a constant (zero-variance) context
            ok = ((np.isfinite(ctx).sum(1) >= self.min_context // 2) & (np.isfinite(tgt).sum(1) >= 1)
                  & (np.nan_to_num(spread, nan=0.0) > 0))
            if not ok.any():
                continue
            ctx, tgt = ctx[ok], tgt[ok]
            m = ctx.shape[0]
            c = np.full((m, self.L), np.nan, np.float32)
            c[:, self.L - ctx.shape[1]:] = ctx
            y = np.full((m, self.H), np.nan, np.float32)
            y[:, :tgt.shape[1]] = tgt
            ctxs.append(c)
            tgts.append(y)
            freqs.append(np.full(m, p.bucket, np.int64))
            groups.append(np.full(m, g, np.int64))
            n += m
            g += 1
        return {
            "context": np.concatenate(ctxs),
            "target": np.concatenate(tgts),
            "freq": np.concatenate(freqs),
            "group": np.concatenate(groups),
        }


def synthetic_panels(n: int = 32, length: int = 3000, seed: int = 0) -> list[Panel]:
    """Small synthetic corpus (trend + seasonality + noise) for tests and smoke runs."""
    rng = np.random.default_rng(seed)
    panels = []
    freqs = ["H", "D", "15T", "W"]
    periods = {"H": 24, "D": 7, "15T": 96, "W": 52}
    for k in range(n):
        f = freqs[k % len(freqs)]
        C = int(rng.integers(1, 5))
        t = np.arange(length)
        base = rng.normal(0, 1) + 0.001 * rng.normal() * t
        vals = []
        for _ in range(C):
            amp = rng.uniform(0.5, 2.0)
            phase = rng.uniform(0, 2 * np.pi)
            vals.append(base + amp * np.sin(2 * np.pi * t / periods[f] + phase) + 0.2 * rng.normal(size=length))
        panels.append(Panel(f"synthetic_{f}", f, np.asarray(vals, np.float32)))
    return panels
