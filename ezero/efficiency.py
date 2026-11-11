"""Inference-cost measurement (thesis Sec. 4.4, Table 3-4, Eq. 3/27).

For every forecaster we time one batch of ``batch_size`` univariate series with a
512-step context and a 96-step horizon:

* GPU TIME (ms)  - mean wall time per batch on the GPU (after warm-up, synchronised);
* MEM (GB)       - peak CUDA memory allocated during the batch;
* CPU TIME (s)   - mean wall time per batch with the model on the CPU;
* Params (M)     - number of parameters, when the wrapper exposes the torch module.

Efficiency score (Eq. 27 with w = 1): mean over the available terms of
``value / max(value over models)`` for memory, GPU time and CPU time. GPU utilisation is
not measured reliably on shared Kaggle GPUs, so it is replaced by the CPU-time term;
lower is better.
"""
from __future__ import annotations

import time

import numpy as np


def count_params(forecaster) -> float | None:
    mod = getattr(forecaster, "torch_module", None)
    if callable(mod):
        mod = mod()
    if mod is None:
        return None
    return sum(p.numel() for p in mod.parameters()) / 1e6


def _time(fn, reps: int, sync=None):
    fn()  # warm-up
    if sync:
        sync()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        t.append(time.perf_counter() - t0)
    return float(np.median(t))


def measure(forecaster, batch_size: int = 32, context_length: int = 512, horizon: int = 96,
            reps: int = 10, cpu_reps: int = 3, freq: str = "H", devices=("cuda", "cpu"), seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    t = np.arange(context_length)
    ctx = (np.sin(2 * np.pi * t / 24)[None, :, None] + 0.1 * rng.normal(size=(batch_size, context_length, 1)))
    ctx = ctx.astype(np.float32)
    out = dict(model=forecaster.name, batch_size=batch_size, context_length=context_length, horizon=horizon,
               params_m=count_params(forecaster))
    # A full-shot model builds its network on the first predict() call (no training is needed to
    # time inference), so its parameters can only be counted after the timing runs below.
    for dev in devices:
        if dev == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    continue
            except ImportError:
                continue
            if hasattr(forecaster, "to_device"):
                forecaster.to_device("cuda")
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            sec = _time(lambda: forecaster.predict(ctx, freq, horizon), reps, torch.cuda.synchronize)
            peak = torch.cuda.max_memory_allocated()
            out["gpu_time_ms"] = round(1000 * sec, 3)
            out["gpu_mem_gb"] = round(max(peak, base) / 1e9, 4)
        elif dev == "cpu":
            if hasattr(forecaster, "to_device"):
                forecaster.to_device("cpu")
            elif getattr(forecaster, "gpu_only", False):
                continue
            sec = _time(lambda: forecaster.predict(ctx, freq, horizon), cpu_reps)
            out["cpu_time_s"] = round(sec, 4)
    if out.get("params_m") is None:
        out["params_m"] = count_params(forecaster)
    return out


def efficiency_scores(rows: list[dict], terms=("gpu_mem_gb", "gpu_time_ms", "cpu_time_s")) -> list[dict]:
    maxima = {k: max((r[k] for r in rows if r.get(k) is not None and np.isfinite(r[k])), default=None)
              for k in terms}
    for r in rows:
        vals = [r[k] / maxima[k] for k in terms
                if r.get(k) is not None and maxima[k] and np.isfinite(r[k])]
        r["efficiency_score"] = round(float(np.mean(vals)), 4) if vals else None
    return rows
