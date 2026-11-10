#!/usr/bin/env python3
"""
dev_eval.py - score E-ZERO variants on the zero-shot development benchmark (ezero/devsuite.py).

Tweaks are chosen here, never on the seven evaluation datasets. Results are appended to
results/dev/dev_results.csv (one row per forecaster x dataset, plus the relative score).

Usage:
  python benchmarks/dev_eval.py --checkpoint data/models/ezero_base.pt --tta none flip multictx flip+multictx
  python benchmarks/dev_eval.py --checkpoint data/models/ezero_v3_*.pt --label v3-candidates
"""
import argparse
import datetime as dt
import glob
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import pandas as pd  # noqa: E402

from baselines.ezero_forecaster import EZeroForecaster  # noqa: E402
from baselines.naive import MeanForecaster, NaiveForecaster  # noqa: E402
from ezero.devsuite import evaluate_dev, load_dev_series  # noqa: E402

OUT = os.path.join(ROOT, "results", "dev", "dev_results.csv")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", nargs="+", default=[os.path.join(ROOT, "data", "models", "ezero_base.pt")])
    p.add_argument("--tta", nargs="+", default=["none"])
    p.add_argument("--baselines", action="store_true", help="also score Naive and Mean")
    p.add_argument("--lotsa-root", default=os.environ.get("EZERO_LOTSA_DIR", os.path.join(ROOT, "data", "lotsa")))
    p.add_argument("--label", default="")
    a = p.parse_args(argv)
    series = load_dev_series(a.lotsa_root)
    print(f"[dev] {len(series)} series from {sorted({s[0] for s in series})}")
    cands = []
    for ck in [c for pat in a.checkpoint for c in sorted(glob.glob(pat))]:
        for t in a.tta:
            cands.append((f"{os.path.basename(ck)}|tta={t}", EZeroForecaster(ck, tta=t)))
    if a.baselines:
        cands += [("Naive", NaiveForecaster()), ("Mean", MeanForecaster())]
    rows = []
    for name, f in cands:
        r = evaluate_dev(f, series)
        print(f"[dev] {name:45s} score={r['score']:.4f}  " +
              " ".join(f"{k[:8]}={v['rel']:.3f}" for k, v in r["datasets"].items()), flush=True)
        for k, v in r["datasets"].items():
            rows.append(dict(candidate=name, dataset=k, mse=v["mse"], snaive_mse=v["snaive"], rel=v["rel"],
                             score=r["score"], label=a.label,
                             time=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
