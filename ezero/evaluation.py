"""Zero-shot / full-shot evaluation on the LSF benchmarks with streaming metrics.

A *forecaster* implements::

    predict(context: np.ndarray (W, L, C), freq: str, horizon: int) -> np.ndarray (W, horizon, C)

on standard-scaled data. Attributes used here:

* ``name``             - registry id
* ``multi_horizon``    - True if forecasting ``max(horizons)`` once and taking prefixes is valid
                         (E-ZERO, naive, most foundation models). False -> one call per horizon.
* ``fit(data, horizon)`` (optional) - called before evaluation for full-shot baselines.

Metrics are MSE / MAE on the scaled values, averaged over all (window, step, channel)
entries, as in the Informer / Time-Series-Library protocol.
"""
from __future__ import annotations

import time

import numpy as np

from ezero.data.ltsf import LSFData, drop_last_batch, window_starts


def _gather(scaled: np.ndarray, starts_abs: np.ndarray, length: int, end: int):
    idx = starts_abs[:, None] + np.arange(length)[None, :]
    valid = idx < end
    vals = scaled[np.minimum(idx, len(scaled) - 1)]
    return vals, valid


def evaluate_forecaster(forecaster, data: LSFData, horizons=(96, 192, 336, 720), protocol: str = "full",
                        batch_series: int = 4096, max_batches: int | None = None, verbose: bool = False) -> list[dict]:
    scaled = data.scaled()
    s, e = data.borders["test"]
    L = data.context_length
    C = data.channels
    horizons = sorted(int(h) for h in horizons)
    starts = window_starts(data, protocol, horizon_max=max(horizons), horizon_min=min(horizons))
    wpb = max(1, batch_series // C)
    multi = getattr(forecaster, "multi_horizon", True)
    passes = [max(horizons)] if multi else horizons
    stats = {h: [0.0, 0.0, 0] for h in horizons}  # sse, sae, n_windows
    B = drop_last_batch(protocol)
    keep_upto = {h: (data.n_windows(h) - data.n_windows(h) % B) if B else None for h in horizons}
    pred_time = 0.0
    n_series = 0
    for ph in passes:
        targets_for = horizons if multi else [ph]
        for bi, k in enumerate(range(0, len(starts), wpb)):
            if max_batches is not None and bi >= max_batches:
                break
            st = starts[k:k + wpb]
            ctx, _ = _gather(scaled, s + st, L, e)                       # (W, L, C)
            t0 = time.perf_counter()
            pred = np.asarray(forecaster.predict(ctx.astype(np.float32), data.freq, ph), dtype=np.float64)
            pred_time += time.perf_counter() - t0
            if ph == passes[0]:
                n_series += ctx.shape[0] * C
            tgt, valid = _gather(scaled, s + st + L, ph, e)              # (W, ph, C)
            for h in targets_for:
                ok = valid[:, h - 1]                                     # window has a full target of length h
                if keep_upto[h] is not None:                             # emulate drop_last=True
                    ok = ok & (st < keep_upto[h])
                if protocol == "last":                                   # one window: the last of horizon h
                    ok = ok & (st == data.n_windows(h) - 1)
                if not ok.any():
                    continue
                d = pred[ok, :h, :] - tgt[ok, :h, :]
                stats[h][0] += float(np.square(d).sum())
                stats[h][1] += float(np.abs(d).sum())
                stats[h][2] += int(ok.sum())
            if verbose:
                print(f"  [{forecaster.name}] {data.name} h={ph} batch {bi + 1}", flush=True)
    rows = []
    for h in horizons:
        sse, sae, nw = stats[h]
        denom = max(1, nw * h * C)
        rows.append(dict(dataset=data.name, horizon=h, mse=sse / denom, mae=sae / denom, n_windows=nw,
                         n_channels=C, protocol=protocol, scaler=data.scaler, context_length=L,
                         predict_seconds=round(pred_time, 3), n_series=n_series))
    return rows


def summarize(rows: list[dict]) -> dict:
    """Average MSE/MAE over horizons per dataset (thesis Table 2 convention)."""
    out = {}
    for r in rows:
        out.setdefault(r["dataset"], []).append(r)
    return {d: dict(mse=float(np.mean([x["mse"] for x in rs])), mae=float(np.mean([x["mae"] for x in rs])))
            for d, rs in out.items()}
