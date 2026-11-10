#!/usr/bin/env python3
"""
fetch_outputs.py - download finished Kaggle kernel outputs back into the repository.

Usage:
  python kaggle/fetch_outputs.py --user <kaggle-username>            # all stages
  python kaggle/fetch_outputs.py --user <kaggle-username> --stages core ablations
  python kaggle/fetch_outputs.py --status --user <kaggle-username>   # only print kernel status

Artifacts are copied from outputs/<repo path> into the same repo path (results/cache,
results/training_logs, data/models); logs go to results/kaggle_logs/<stage>/. Afterwards run
``python benchmarks/run_benchmarks.py --merge-only`` and the figure scripts.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from build_kernels import STAGES  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--user", default=os.environ.get("KAGGLE_USERNAME"))
    p.add_argument("--stages", nargs="*", default=list(STAGES))
    p.add_argument("--status", action="store_true")
    args = p.parse_args(argv)
    if not args.user:
        p.error("--user is required")
    for s in args.stages:
        ref = f"{args.user}/ezero-{s}"
        subprocess.call(["kaggle", "kernels", "status", ref])
        if args.status:
            continue
        tmp = tempfile.mkdtemp(prefix=f"ezero-{s}-")
        rc = subprocess.call(["kaggle", "kernels", "output", ref, "-p", tmp])
        if rc:
            print(f"[fetch] {ref}: output download failed ({rc})")
            continue
        n = 0
        out = os.path.join(tmp, "outputs")
        for root, _, files in os.walk(out):
            for f in files:
                src = os.path.join(root, f)
                dst = os.path.join(ROOT, os.path.relpath(src, out))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                n += 1
        logdir = os.path.join(ROOT, "results", "kaggle_logs", s)
        os.makedirs(logdir, exist_ok=True)
        for f in ("stage_status.json",):
            if os.path.exists(os.path.join(tmp, f)):
                shutil.copy2(os.path.join(tmp, f), logdir)
        if os.path.isdir(os.path.join(tmp, "logs")):
            for f in os.listdir(os.path.join(tmp, "logs")):
                shutil.copy2(os.path.join(tmp, "logs", f), logdir)
        print(f"[fetch] {ref}: {n} artifacts copied into the repo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
