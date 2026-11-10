#!/usr/bin/env python3
"""
run_benchmarks.py - Unified benchmark CLI (same workflow as gym-invmgmt / Paper 1).

Evaluate one model at a time across all LSF datasets and horizons, with per-model caches
(results/cache/bench_<model>.csv) and resumability, then merge everything into the
canonical results/benchmark_final_merged.csv consumed by the figure/table scripts.

Usage:
  python benchmarks/run_benchmarks.py --list
  python benchmarks/run_benchmarks.py --model E-ZERO --protocol full
  python benchmarks/run_benchmarks.py --model E-ZERO --protocol sub16
  python benchmarks/run_benchmarks.py --model group:naive --protocol full
  python benchmarks/run_benchmarks.py --model group:ezero-ablation --protocol full
  python benchmarks/run_benchmarks.py --model Chronos-Bolt-T --protocol sub16
  python benchmarks/run_benchmarks.py --model DLinear --protocol full --fullshot-max-steps 3000
  python benchmarks/run_benchmarks.py --model ALL --protocol sub16 --skip-unavailable
  python benchmarks/run_benchmarks.py --merge-only
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import subprocess
import sys
import time
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ezero.data.ltsf import DEFAULT_DATASETS, DEFAULT_HORIZONS, LSF_DATASETS, load_lsf  # noqa: E402
from ezero.evaluation import evaluate_forecaster  # noqa: E402

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")
MODELS_DIR = os.path.join(PROJECT_ROOT, "data", "models")
MERGED_CSV = os.path.join(RESULTS_DIR, "benchmark_final_merged.csv")
LONG_CSV = os.path.join(RESULTS_DIR, "benchmark_long.csv")
MANIFEST = os.path.join(RESULTS_DIR, "manifest.json")

# ---------------------------------------------------------------------------
# Model registry (the analogue of Paper 1's agent roster)
# ---------------------------------------------------------------------------
EZERO_VARIANTS = {
    "E-ZERO": "base",
    "E-ZERO-noMRC": "noMRC",
    "E-ZERO-noDTP": "noDTP",
    "E-ZERO-noRAE": "noRAE",
    "E-ZERO-noCRRB": "noCRRB",
    "E-ZERO-noExog": "noExog",
    "E-ZERO-noTCN": "noTCN",
    "E-ZERO-noNBEATS": "noNBEATS",
}

MODEL_REGISTRY: dict[str, dict] = {}
for _mid, _var in EZERO_VARIANTS.items():
    MODEL_REGISTRY[_mid] = dict(group="ezero" if _var == "base" else "ezero-ablation", variant=_var,
                                description=f"E-ZERO pre-trained on LOTSA ({_var})")
MODEL_REGISTRY.update({
    "Naive": dict(group="naive", description="last value"),
    "SeasonalNaive": dict(group="naive", description="repeat last season"),
    "Mean": dict(group="naive", description="context mean"),
    # zero-shot foundation models. era: "thesis" = public before the thesis (Aug/Sep 2024),
    # "post-thesis" = released later (HuggingFace creation date in the description)
    "Chronos-T": dict(group="tsfm", era="thesis", description="amazon/chronos-t5-tiny (2024-02)",
                      package="chronos-forecasting"),
    "TTM-R1": dict(group="tsfm", era="thesis", description="ibm-granite/granite-timeseries-ttm-r1 512-96, rolled (2024-04)",
                   package="granite-tsfm"),
    "Moirai-S": dict(group="tsfm", era="thesis", description="Salesforce/moirai-1.0-R-small (2024-02)", package="uni2ts"),
    "TimesFM-1.0": dict(group="tsfm", era="thesis", description="google/timesfm-1.0-200m-pytorch (2024-08)",
                        package="timesfm<2 (Python 3.11)"),
    "Lag-Llama": dict(group="tsfm", era="thesis", description="time-series-foundation-models/Lag-Llama (2024-02)",
                      package="lag-llama GitHub code (Python 3.11, opt-in)", opt_in=True),
    "Chronos-Bolt-T": dict(group="tsfm", era="post-thesis", description="amazon/chronos-bolt-tiny (2024-11)",
                           package="chronos-forecasting"),
    "TTM-R2": dict(group="tsfm", era="post-thesis", description="ibm-granite/granite-timeseries-ttm-r2 (2024-10)",
                   package="granite-tsfm"),
    "TimesFM-2.5": dict(group="tsfm", era="post-thesis", description="google/timesfm-2.5-200m-pytorch (2025-09)",
                        package="timesfm>=2"),
    # thesis Table 3/4 only: not zero-shot forecasters (need a trained head) -> timed, never scored
    "GPT4TS": dict(group="efficiency", era="thesis", accuracy=False,
                   description="One Fits All: GPT-2 (6 layers) + patching, untrained head (timing only)",
                   package="transformers"),
    "MOMENT": dict(group="efficiency", era="thesis", accuracy=False,
                   description="AutonLab/MOMENT-1-large forecasting head (timing only, 2024-05)",
                   package="momentfm (Python 3.11)"),
    "DLinear": dict(group="fullshot", description="DLinear trained per dataset/horizon"),
    "PatchTST": dict(group="fullshot", description="PatchTST trained per dataset/horizon"),
    "iTransformer": dict(group="fullshot", description="iTransformer trained per dataset/horizon"),
    "TSMixer": dict(group="fullshot", description="TSMixer trained per dataset/horizon"),
    "TimeMixer": dict(group="fullshot", description="TimeMixer trained per dataset/horizon"),
    "TimesNet": dict(group="fullshot", description="TimesNet trained per dataset/horizon"),
    "FEDformer": dict(group="fullshot", description="FEDformer (Fourier) trained per dataset/horizon"),
    "Autoformer": dict(group="fullshot", description="Autoformer trained per dataset/horizon"),
})
ALL_MODEL_IDS = list(MODEL_REGISTRY)
GROUPS = sorted({v["group"] for v in MODEL_REGISTRY.values()})


def checkpoint_path(model_id: str, models_dir: str = MODELS_DIR) -> str:
    return os.path.join(models_dir, f"ezero_{MODEL_REGISTRY[model_id]['variant']}.pt")


def resolve_models(spec: str) -> list[str]:
    if spec == "ALL":
        return list(ALL_MODEL_IDS)
    out = []
    for part in spec.split(","):
        part = part.strip()
        if part.startswith("group:"):
            g = part.split(":", 1)[1]
            out += [m for m in ALL_MODEL_IDS if MODEL_REGISTRY[m]["group"] == g]
        elif part in MODEL_REGISTRY:
            out.append(part)
        else:
            raise SystemExit(f"unknown model {part!r}; see --list")
    return out


def make_forecaster(model_id: str, args) -> object:
    meta = MODEL_REGISTRY[model_id]
    g = meta["group"]
    if g in ("ezero", "ezero-ablation"):
        from baselines.ezero_forecaster import EZeroForecaster
        ckpt = checkpoint_path(model_id, args.models_dir)
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"{ckpt} missing - run benchmarks/pretrain.py --variant {meta['variant']}")
        return EZeroForecaster(ckpt, name=model_id, batch_series=args.batch_series)
    if g == "naive":
        from baselines import naive
        return {"Naive": naive.NaiveForecaster, "SeasonalNaive": naive.SeasonalNaiveForecaster,
                "Mean": naive.MeanForecaster}[model_id]()
    if g == "tsfm":
        from baselines.tsfm import TSFM_FACTORIES
        return TSFM_FACTORIES[model_id]()
    if g == "efficiency":
        from baselines.efficiency_only import EFFICIENCY_ONLY_FACTORIES
        return EFFICIENCY_ONLY_FACTORIES[model_id]()
    if g == "fullshot":
        from baselines.fullshot import FullShotForecaster
        return FullShotForecaster(model_id, max_steps=args.fullshot_max_steps, max_epochs=args.fullshot_max_epochs)
    raise ValueError(model_id)


# ---------------------------------------------------------------------------
# Cache handling
# ---------------------------------------------------------------------------
def _safe(model_id: str) -> str:
    return model_id.replace("/", "_")


def cache_path(model_id: str) -> str:
    return os.path.join(CACHE_DIR, f"bench_{_safe(model_id)}.csv")


def load_cache(model_id: str) -> pd.DataFrame:
    p = cache_path(model_id)
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def _done(df: pd.DataFrame, dataset: str, protocol: str, scaler: str, horizons) -> bool:
    if df.empty:
        return False
    m = (df["dataset"] == dataset) & (df["protocol"] == protocol) & (df["scaler"] == scaler)
    return set(int(h) for h in horizons) <= set(df.loc[m, "horizon"].astype(int))


def append_rows(model_id: str, rows: list[dict]):
    os.makedirs(CACHE_DIR, exist_ok=True)
    new = pd.DataFrame(rows)
    old = load_cache(model_id)
    if not old.empty:
        key = ["dataset", "horizon", "protocol", "scaler"]
        merged = pd.concat([old, new], ignore_index=True)
        merged = merged.drop_duplicates(subset=key, keep="last")
    else:
        merged = new
    merged.to_csv(cache_path(model_id), index=False)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", PROJECT_ROOT, "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return os.environ.get("EZERO_GIT_COMMIT")


def device_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return platform.processor() or "cpu"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def eval_one_model(model_id: str, args) -> list[dict]:
    if MODEL_REGISTRY[model_id].get("accuracy") is False:
        print(f"[{model_id}] efficiency-only model (needs a trained head); use benchmarks/run_efficiency.py")
        return []
    datasets = args.datasets
    horizons = args.horizons
    cache = load_cache(model_id)
    todo = [d for d in datasets if args.force or not _done(cache, d, args.protocol, args.scaler, horizons)]
    if not todo:
        print(f"[{model_id}] all {len(datasets)} datasets cached for protocol={args.protocol} scaler={args.scaler}")
        return []
    t_load = time.time()
    forecaster = make_forecaster(model_id, args)
    load_s = time.time() - t_load
    info_base = forecaster.info() if hasattr(forecaster, "info") else {}
    all_rows = []
    for ds in todo:
        data = load_lsf(ds, context_length=512, scaler=args.scaler)
        t0 = time.time()
        rows = []
        if getattr(forecaster, "zero_shot", True):
            rows = evaluate_forecaster(forecaster, data, horizons, args.protocol, args.batch_series,
                                       max_batches=args.max_batches)
        else:  # full-shot: train one model per horizon
            for h in horizons:
                forecaster.fit(data, h)
                r = evaluate_forecaster(forecaster, data, [h], args.protocol, args.batch_series,
                                        max_batches=args.max_batches)[0]
                r.update({f"fit_{k}": v for k, v in forecaster.info().items()})
                rows.append(r)
        info = dict(info_base)
        info.pop("config", None)
        for r in rows:
            r.update(model=model_id, group=MODEL_REGISTRY[model_id]["group"],
                     zero_shot=bool(getattr(forecaster, "zero_shot", True)), wall_seconds=round(time.time() - t0, 2),
                     load_seconds=round(load_s, 2), device=device_name(), git_commit=git_commit(),
                     timestamp=_now(),
                     smoke=args.max_batches is not None,
                     **{f"info_{k}": (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in info.items()})
        append_rows(model_id, rows)
        all_rows += rows
        avg = np.mean([r["mse"] for r in rows])
        print(f"[{model_id}] {ds:12s} protocol={args.protocol} avg MSE={avg:.4f} "
              f"({time.time() - t0:.1f}s)", flush=True)
    return all_rows


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
def merge_all(include_smoke: bool = False) -> pd.DataFrame:
    frames = []
    for mid in ALL_MODEL_IDS:
        df = load_cache(mid)
        if df.empty:
            continue
        if not include_smoke and "smoke" in df.columns:
            df = df[~df["smoke"].astype(bool)]
        frames.append(df)
    if not frames:
        print("[merge] no caches found")
        return pd.DataFrame()
    long = pd.concat(frames, ignore_index=True)
    long = long.sort_values(["protocol", "scaler", "dataset", "horizon", "model"])
    long.to_csv(LONG_CSV, index=False)

    # wide table: one row per (protocol, scaler, dataset, horizon) + 'avg' rows
    rows = []
    order = [m for m in ALL_MODEL_IDS if m in set(long["model"])]
    for (prot, sc, ds), g in long.groupby(["protocol", "scaler", "dataset"], sort=False):
        horizons = sorted(g["horizon"].astype(int).unique())
        for h in horizons + ["avg"]:
            row = {"Protocol": prot, "Scaler": sc, "Dataset": ds, "Horizon": h}
            for m in order:
                gm = g[g["model"] == m]
                if gm.empty:
                    continue
                if h == "avg":
                    # average only when the model covers the four canonical horizons
                    if set(gm["horizon"].astype(int)) >= set(DEFAULT_HORIZONS):
                        sel = gm[gm["horizon"].astype(int).isin(DEFAULT_HORIZONS)]
                        row[f"{m}_MSE"] = sel["mse"].mean()
                        row[f"{m}_MAE"] = sel["mae"].mean()
                else:
                    r = gm[gm["horizon"].astype(int) == h]
                    if not r.empty:
                        row[f"{m}_MSE"] = float(r["mse"].iloc[0])
                        row[f"{m}_MAE"] = float(r["mae"].iloc[0])
            rows.append(row)
    wide = pd.DataFrame(rows)
    ds_order = {d: i for i, d in enumerate(DEFAULT_DATASETS)}
    wide["_d"] = wide["Dataset"].map(ds_order)
    wide["_h"] = wide["Horizon"].map(lambda h: 10_000 if h == "avg" else int(h))
    wide = wide.sort_values(["Protocol", "Scaler", "_d", "_h"]).drop(columns=["_d", "_h"])
    wide.to_csv(MERGED_CSV, index=False)
    write_manifest(long)
    print(f"[merge] {len(long)} long rows from {long['model'].nunique()} models -> {MERGED_CSV} ({len(wide)} rows)")
    return wide


def write_manifest(long: pd.DataFrame):
    from ezero.checkpoint import file_sha256
    ckpts = {}
    for mid, var in EZERO_VARIANTS.items():
        p = os.path.join(MODELS_DIR, f"ezero_{var}.pt")
        if os.path.exists(p):
            ckpts[mid] = {"path": os.path.relpath(p, PROJECT_ROOT), "bytes": os.path.getsize(p),
                          "sha256": file_sha256(p)}
    coverage = {}
    for (prot, sc), g in long.groupby(["protocol", "scaler"]):
        coverage[f"{prot}/{sc}"] = {m: sorted(gm["dataset"].unique().tolist()) for m, gm in g.groupby("model")}
    eff = os.path.join(RESULTS_DIR, "efficiency.csv")
    manifest = {
        "generated_at": _now(),
        "git_commit": git_commit(),
        "model_roster_count": len(ALL_MODEL_IDS),
        "models": ALL_MODEL_IDS,
        "groups": {g: [m for m in ALL_MODEL_IDS if MODEL_REGISTRY[m]["group"] == g] for g in GROUPS},
        "datasets": DEFAULT_DATASETS,
        "horizons": list(DEFAULT_HORIZONS),
        "dataset_sha256": {k: v["sha256"] for k, v in LSF_DATASETS.items()},
        "canonical_csv_path": os.path.relpath(MERGED_CSV, PROJECT_ROOT),
        "long_csv_path": os.path.relpath(LONG_CSV, PROJECT_ROOT),
        "efficiency_csv_path": os.path.relpath(eff, PROJECT_ROOT) if os.path.exists(eff) else None,
        "coverage": coverage,
        "checkpoints": ckpts,
        "devices": sorted(long["device"].dropna().unique().tolist()) if "device" in long else [],
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)


def list_models():
    print(f"{len(ALL_MODEL_IDS)} registered models:")
    for g in GROUPS:
        print(f"  [{g}]")
        for m in ALL_MODEL_IDS:
            if MODEL_REGISTRY[m]["group"] == g:
                extra = f" (pip: {MODEL_REGISTRY[m]['package']})" if "package" in MODEL_REGISTRY[m] else ""
                print(f"    {m:18s} {MODEL_REGISTRY[m]['description']}{extra}")


def main(argv=None):
    p = argparse.ArgumentParser(description="E-ZERO unified benchmark runner")
    p.add_argument("--model", type=str, default=None, help="model id, comma list, group:<g> or ALL")
    p.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    p.add_argument("--horizons", nargs="*", type=int, default=list(DEFAULT_HORIZONS))
    p.add_argument("--protocol", default="full", help="full | subN (e.g. sub16) | strideN")
    p.add_argument("--scaler", default="train", choices=["train", "full"])
    p.add_argument("--batch-series", type=int, default=4096)
    p.add_argument("--max-batches", type=int, default=None, help="smoke test: limit batches per dataset")
    p.add_argument("--models-dir", default=MODELS_DIR)
    p.add_argument("--fullshot-max-steps", type=int, default=3000)
    p.add_argument("--fullshot-max-epochs", type=int, default=10)
    p.add_argument("--force", action="store_true", help="re-run even if cached")
    p.add_argument("--skip-unavailable", action="store_true", help="continue when a model cannot be loaded")
    p.add_argument("--list", action="store_true")
    p.add_argument("--merge-only", action="store_true")
    p.add_argument("--no-merge", action="store_true")
    args = p.parse_args(argv)

    if args.list:
        list_models()
        return 0
    if args.merge_only:
        merge_all()
        return 0
    if not args.model:
        p.error("--model is required (or use --list / --merge-only)")
    unknown = [d for d in args.datasets if d not in LSF_DATASETS]
    if unknown:
        p.error(f"unknown datasets {unknown}")
    failures = {}
    for mid in resolve_models(args.model):
        try:
            eval_one_model(mid, args)
        except Exception as exc:
            failures[mid] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            if not args.skip_unavailable:
                break
    if not args.no_merge:
        merge_all()
    if failures:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, "failures.json"), "a") as f:
            f.write(json.dumps({"time": _now(), "protocol": args.protocol,
                                "failures": failures}) + "\n")
        print(f"[run_benchmarks] failures: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
