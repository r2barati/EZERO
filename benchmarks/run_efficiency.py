#!/usr/bin/env python3
"""
run_efficiency.py - inference cost per model (thesis Table 3) and efficiency score (Table 4).

Usage:
  python benchmarks/run_efficiency.py --model E-ZERO,Naive,DLinear
  python benchmarks/run_efficiency.py --model group:tsfm --skip-unavailable
  python benchmarks/run_efficiency.py --merge-only      # recompute scores over all cached rows

Rows are cached per model in results/cache/efficiency_<model>.csv; the merged table with
efficiency scores (normalised over all models present) is results/efficiency.csv.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd  # noqa: E402

from benchmarks.run_benchmarks import (CACHE_DIR, MODELS_DIR, RESULTS_DIR, device_name, make_forecaster,  # noqa: E402
                                       resolve_models)
from ezero.efficiency import efficiency_scores, measure  # noqa: E402

EFF_CSV = os.path.join(RESULTS_DIR, "efficiency.csv")


def merge():
    rows = []
    for p in sorted(glob.glob(os.path.join(CACHE_DIR, "efficiency_*.csv"))):
        rows += pd.read_csv(p).to_dict("records")
    if not rows:
        return None
    rows = [{k: (None if (isinstance(v, float) and v != v) else v) for k, v in r.items()} for r in rows]
    df = pd.DataFrame(efficiency_scores(rows)).sort_values("efficiency_score", na_position="last")
    df.to_csv(EFF_CSV, index=False)
    print(df.to_string(index=False))
    return df


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="E-ZERO")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--cpu-reps", type=int, default=3)
    p.add_argument("--no-cpu", action="store_true")
    p.add_argument("--models-dir", default=MODELS_DIR)
    p.add_argument("--context-length", type=int, default=512, help="lookback the models are timed at")
    p.add_argument("--skip-unavailable", action="store_true")
    p.add_argument("--merge-only", action="store_true")
    args = p.parse_args(argv)
    if args.merge_only:
        merge()
        return 0
    # make_forecaster() reads these; full-shot models build their network on the first predict()
    # call, so no training budget is needed to time inference.
    args.batch_series = 4096
    args.fullshot_max_steps = 0
    args.fullshot_max_epochs = 0
    os.makedirs(CACHE_DIR, exist_ok=True)
    status = 0
    for mid in resolve_models(args.model):
        try:
            fc = make_forecaster(mid, args)
            devices = ("cuda",) if args.no_cpu else ("cuda", "cpu")
            row = measure(fc, args.batch_size, args.context_length, args.horizon, args.reps, args.cpu_reps,
                          devices=devices)
            row["device"] = device_name()
            pd.DataFrame([row]).to_csv(os.path.join(CACHE_DIR, f"efficiency_{mid}.csv"), index=False)
            print(row, flush=True)
        except Exception:
            traceback.print_exc()
            status = 1
            if not args.skip_unavailable:
                break
    merge()
    return status


if __name__ == "__main__":
    sys.exit(main())
