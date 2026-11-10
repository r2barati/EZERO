#!/usr/bin/env python3
"""
verify_publication_ready.py - the final "red button" check (same role as in Paper 1).

1. runs the pytest suite;
2. checks that every registered model has cached results for the canonical protocols
   (E-ZERO + naive + ablations + full-shot on ``full``; E-ZERO + foundation models on ``sub16``);
3. checks checkpoints against the SHA-256 values in results/manifest.json;
4. regenerates tables and figures.

Usage:
  python benchmarks/verify_publication_ready.py            # everything
  python benchmarks/verify_publication_ready.py --quick    # skip pytest
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402

from benchmarks.run_benchmarks import ALL_MODEL_IDS, LONG_CSV, MANIFEST, MODEL_REGISTRY  # noqa: E402
from ezero.checkpoint import file_sha256  # noqa: E402
from ezero.data.ltsf import DEFAULT_DATASETS, DEFAULT_HORIZONS  # noqa: E402

EXPECTED = {
    ("full", "train"): [m for m in ALL_MODEL_IDS if MODEL_REGISTRY[m]["group"] in
                        ("ezero", "ezero-ablation", "naive", "fullshot")],
    ("sub16", "train"): ["E-ZERO"] + [m for m in ALL_MODEL_IDS if MODEL_REGISTRY[m]["group"] in ("tsfm", "naive")],
}


def coverage_report() -> bool:
    if not os.path.exists(LONG_CSV):
        print("[verify] results/benchmark_long.csv missing")
        return False
    long = pd.read_csv(LONG_CSV)
    ok = True
    for (prot, sc), models in EXPECTED.items():
        d = long[(long["protocol"] == prot) & (long["scaler"] == sc)]
        for m in models:
            dm = d[d["model"] == m]
            missing = [ds for ds in DEFAULT_DATASETS
                       if set(dm[dm["dataset"] == ds]["horizon"].astype(int)) < set(DEFAULT_HORIZONS)]
            status = "OK" if not missing else f"MISSING {missing}"
            ok &= not missing
            print(f"[verify] {prot:6s} {m:18s} {status}")
    return ok


def checkpoint_report() -> bool:
    if not os.path.exists(MANIFEST):
        print("[verify] manifest missing")
        return False
    man = json.load(open(MANIFEST))
    ok = True
    for mid, info in man.get("checkpoints", {}).items():
        p = os.path.join(ROOT, info["path"])
        good = os.path.exists(p) and file_sha256(p) == info["sha256"]
        ok &= good
        print(f"[verify] checkpoint {mid:18s} {'OK' if good else 'MISMATCH'}")
    return ok


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args(argv)
    ok = True
    if not args.quick:
        ok &= subprocess.call([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT) == 0
    subprocess.call([sys.executable, os.path.join(ROOT, "benchmarks", "run_benchmarks.py"), "--merge-only"])
    ok &= coverage_report()
    ok &= checkpoint_report()
    for s in ("extract_tables.py", "regenerate_figures.py"):
        ok &= subprocess.call([sys.executable, os.path.join(ROOT, "paper", "figures", s)]) == 0
    print("\n[verify] PUBLICATION READY" if ok else "\n[verify] NOT READY - see messages above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
