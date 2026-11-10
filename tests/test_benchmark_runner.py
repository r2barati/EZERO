"""Registry, evaluation and merge contracts of benchmarks/run_benchmarks.py."""
import numpy as np

from baselines.naive import NaiveForecaster, SeasonalNaiveForecaster
from benchmarks import run_benchmarks as rb
from ezero.data.ltsf import LSFData, compute_borders
from ezero.evaluation import evaluate_forecaster


def toy_data(T=4000, C=3, L=64):
    t = np.arange(T)
    v = np.stack([np.sin(2 * np.pi * t / 24 + c) for c in range(C)], 1).astype(np.float32)
    b = compute_borders(T, "ratio", L)
    return LSFData("toy", "H", v, b, v.mean(0), v.std(0), "train", L)


def test_roster():
    assert len(rb.ALL_MODEL_IDS) == 24
    assert [m for m in rb.ALL_MODEL_IDS if rb.MODEL_REGISTRY[m].get("accuracy") is False] == ["GPT4TS", "MOMENT"]
    eras = {m: rb.MODEL_REGISTRY[m].get("era") for m in rb.ALL_MODEL_IDS if rb.MODEL_REGISTRY[m]["group"] == "tsfm"}
    assert sorted(m for m, e in eras.items() if e == "thesis") == ["Chronos-T", "Lag-Llama", "Moirai-S",
                                                                   "TTM-R1", "TimesFM-1.0"]
    assert sum(rb.MODEL_REGISTRY[m]["group"] == "ezero-ablation" for m in rb.ALL_MODEL_IDS) == 7
    assert rb.resolve_models("group:naive") == ["Naive", "SeasonalNaive", "Mean"]


def test_seasonal_naive_is_exact_on_periodic_data():
    rows = evaluate_forecaster(SeasonalNaiveForecaster(), toy_data(), [24, 48], "full")
    assert all(r["mse"] < 1e-8 for r in rows)
    assert rows[0]["n_windows"] > rows[1]["n_windows"]


def test_streaming_metrics_match_direct_computation():
    d = toy_data()
    rows = evaluate_forecaster(NaiveForecaster(), d, [24], "full", batch_series=7)
    sc = d.scaled()
    s, e = d.borders["test"]
    errs = []
    for i in range(d.n_windows(24)):
        ctx = sc[s + i: s + i + d.context_length]
        tgt = sc[s + i + d.context_length: s + i + d.context_length + 24]
        errs.append((tgt - ctx[-1]) ** 2)
    assert np.isclose(rows[0]["mse"], np.mean(errs))
