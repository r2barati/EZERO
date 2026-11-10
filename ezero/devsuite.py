"""Zero-shot development benchmark for choosing E-ZERO tweaks without touching the evaluation data.

Model and training choices must not be made on ETTh1/ETTh2/ETTm1/ETTm2/Weather/Electricity/Traffic
(not even their validation splits), otherwise the evaluation is no longer zero-shot. Tweaks are
therefore scored on held-out LOTSA sub-datasets that are neither in the pre-training corpus nor
related to the evaluation data:

    kdd_cup_2018_with_missing (H), pedestrian_counts (H), elecdemand (30T), covid19_energy (H),
    saugeenday (D), us_births (D), sunspot_with_missing (D)

Each univariate series is split chronologically: scaler from the first 70%, test region = last 20%.
For every horizon in {96, 192, 336, 720} that fits, 16 evenly spaced windows (context 512) are scored.
The score of a forecaster is the geometric mean over datasets of MSE / MSE(seasonal naive), averaged
over horizons: < 1 means better than seasonal naive, lower is better.
"""
from __future__ import annotations

import os

import numpy as np

from ezero.data.lotsa import check_no_eval_overlap, download_subset, load_panels

DEV_SUBSETS = ["kdd_cup_2018_with_missing", "pedestrian_counts", "elecdemand", "covid19_energy", "saugeenday",
               "us_births", "sunspot_with_missing"]
HORIZONS = (96, 192, 336, 720)


def load_dev_series(root: str, max_series: int = 64, download: bool = True):
    """List of (dataset, freq, 1-D float32 array) from the dev subsets."""
    check_no_eval_overlap(DEV_SUBSETS)
    out = []
    for name in DEV_SUBSETS:
        path = download_subset(name, root) if download else os.path.join(root, name)
        n = 0
        for p in load_panels(name, path, min_length=512 + 96 + 64):
            for row in p.values:
                if n >= max_series:
                    break
                x = row.astype(np.float32)
                x = x[np.argmax(np.isfinite(x)):] if np.isfinite(x).any() else x
                if len(x) < 512 + 96 + 64:
                    continue
                out.append((name, p.freq, x))
                n += 1
    return out


def _windows(x, L, H, n_win):
    T = len(x)
    test_start = int(T * 0.8)
    first = max(test_start - L, 0)
    last = T - L - H
    if last < first:
        return None
    starts = np.unique(np.linspace(first, last, num=n_win).round().astype(int))
    return starts


def evaluate_dev(forecaster, series, L: int = 512, n_win: int = 16, horizons=HORIZONS) -> dict:
    """Per-dataset mean MSE over horizons, plus seasonal-naive reference and the relative score."""
    from baselines.naive import SeasonalNaiveForecaster
    snaive = SeasonalNaiveForecaster()
    per = {}
    for name, freq, x in series:
        mu = np.nanmean(x[: int(len(x) * 0.7)])
        sd = np.nanstd(x[: int(len(x) * 0.7)]) or 1.0
        z = ((x - mu) / sd).astype(np.float32)
        z = np.where(np.isfinite(z), z, np.nan)
        for H in horizons:
            st = _windows(z, L, H, n_win)
            if st is None:
                continue
            ctx = np.stack([z[s:s + L] for s in st])[:, :, None]
            tgt = np.stack([z[s + L:s + L + H] for s in st])
            ctx_f = np.where(np.isfinite(ctx), ctx, 0.0).astype(np.float32)  # naive needs finite input
            ok = np.isfinite(tgt)
            if ok.sum() == 0:
                continue
            p = np.asarray(forecaster.predict(ctx if getattr(forecaster, "handles_nan", False) else ctx_f, freq, H))[..., 0]
            q = np.asarray(snaive.predict(ctx_f, freq, H))[..., 0]
            e = np.where(ok, (p[:, :H] - np.nan_to_num(tgt)) ** 2, 0).sum() / ok.sum()
            r = np.where(ok, (q[:, :H] - np.nan_to_num(tgt)) ** 2, 0).sum() / ok.sum()
            d = per.setdefault(name, {"mse": [], "snaive": []})
            d["mse"].append(e)
            d["snaive"].append(r)
    rows = {k: dict(mse=float(np.mean(v["mse"])), snaive=float(np.mean(v["snaive"])),
                    rel=float(np.mean(v["mse"]) / max(np.mean(v["snaive"]), 1e-12))) for k, v in per.items()}
    score = float(np.exp(np.mean([np.log(r["rel"]) for r in rows.values()]))) if rows else float("nan")
    return dict(score=score, datasets=rows)
