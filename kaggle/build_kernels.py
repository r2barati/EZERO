#!/usr/bin/env python3
"""
build_kernels.py - render (and optionally push) the Kaggle kernels that run the pipeline.

Usage:
  python kaggle/build_kernels.py --user <kaggle-username>                 # render kaggle/kernels/*
  python kaggle/build_kernels.py --user <kaggle-username> --push core     # render + push one stage
  python kaggle/build_kernels.py --user <kaggle-username> --push ALL

Each kernel is a private GPU (T4) script kernel with internet enabled. It clones this
repository at the commit given by --commit (default: current HEAD, which must be pushed),
so a run is pinned to an exact code version. The LSF CSVs (ETTh1/ETTm1/Weather/Electricity/
Traffic) come from the attached public Kaggle dataset; ETTh2/ETTm2 are fetched from the
official ETDataset GitHub repository; LOTSA and foundation-model weights from HuggingFace.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO_URL = "https://github.com/r2barati/EZERO.git"
LSF_DATASET = "giochelavaipiatti/time-series-forecasts-popular-benchmark-datasets"
STAGES = {
    "core": "E-ZERO pre-training + zero-shot evaluation + naive baselines",
    "ablations": "E-ZERO ablation variants (thesis Table 5)",
    "tsfm": "Zero-shot foundation-model baselines (Chronos, TTM, Moirai, TimesFM)",
    "fullshot": "Full-shot baselines (DLinear, PatchTST, iTransformer)",
    "fullshot2": "Full-shot baselines (TSMixer, TimeMixer, TimesNet)",
    "fullshot3": "Full-shot baselines (FEDformer, Autoformer)",
    "tsfmval": "Foundation models under their papers' protocols (reproduction check)",
    "fullval": "Full-shot baselines with their official settings (reproduction check)",
    "tweaks": "E-ZERO v3 candidates, dev-benchmark selection, single final evaluation",
    "faithful": "Thesis Sec. 3.3.3-3.3.5 components implemented, plus their ablation",
    "tsfmfix": "Moirai-S re-run with median aggregation; Lag-Llama on the affordable datasets",
    "smoke": "Short end-to-end environment check",
}


def head_commit() -> str:
    return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()


def render(stage: str, user: str, commit: str, machine: str, tsfm_models: str = "ALL") -> str:
    slug = f"ezero-{stage}"
    d = os.path.join(HERE, "kernels", slug)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(HERE, "kernel_runner.py")) as f:
        code = f.read()
    code = (code.replace("__REPO_URL__", REPO_URL).replace("__COMMIT__", commit).replace("__STAGE__", stage)
            .replace("__TSFM_MODELS__", tsfm_models))
    with open(os.path.join(d, f"{slug}.py"), "w") as f:
        f.write(code)
    meta = {
        "id": f"{user}/{slug}",
        "title": f"EZERO {stage}",
        "code_file": f"{slug}.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu", "time series"],
        "dataset_sources": [LSF_DATASET],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": machine,
    }
    with open(os.path.join(d, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return d


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--user", default=os.environ.get("KAGGLE_USERNAME"), required=False)
    p.add_argument("--commit", default=None)
    p.add_argument("--machine", default="NvidiaTeslaT4")
    p.add_argument("--stages", nargs="*", default=list(STAGES))
    p.add_argument("--push", nargs="*", default=None, help="stages to push (or ALL)")
    p.add_argument("--tsfm-models", default="ALL", help="tsfm stage: comma list of model ids to run (default ALL)")
    args = p.parse_args(argv)
    if not args.user:
        p.error("--user (Kaggle username) is required")
    commit = args.commit or head_commit()
    dirs = {s: render(s, args.user, commit, args.machine, args.tsfm_models) for s in args.stages}
    for s, d in dirs.items():
        print(f"rendered {s:10s} -> {os.path.relpath(d, ROOT)} @ {commit[:10]}")
    if args.push is not None:
        todo = list(dirs) if args.push in ([], ["ALL"]) else args.push
        for s in todo:
            rc = subprocess.call(["kaggle", "kernels", "push", "-p", dirs[s]])
            if rc:
                print(f"push of {s} failed ({rc})", file=sys.stderr)
                return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
