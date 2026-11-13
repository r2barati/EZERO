#!/usr/bin/env python3
"""
run_stage.py - run one benchmark stage outside Kaggle (Google Colab, or any GPU box).

    python colab/run_stage.py --stage fullshot2 --out /content/drive/MyDrive/ezero-runs

It runs exactly the commands the Kaggle kernel runs: it renders ``kaggle/kernel_runner.py``
with this checkout's remote and commit and executes it, so a run here and a run on Kaggle
produce the same rows from the same code at the same commit.

Two things make it usable on a session that can be cut off at any moment:

* ``--out`` is a directory that survives the session - on Colab, a folder in your Google
  Drive. Everything the stage produces is copied there as it goes.
* Before the stage starts, whatever is already in ``--out`` is copied back into the
  checkout. Every benchmark is cached per model and dataset, so a second run skips the work
  the first one finished. Run the same command again after a disconnect and it carries on.

Stages, and what each still needs (November 2026):

  tsfmfix    Moirai-S re-run after the median fix, and Lag-Llama on the five datasets it can
             finish. ~1 h for Moirai-S, ~6 h for Lag-Llama. Use --tsfm-models to split them.
  tsfmval    the foundation models under the protocols their own papers used.   ~4 h
  fullval    the full-shot baselines with their published per-dataset settings. ~6 h
  fullshot2  TSMixer, TimeMixer and TimesNet trained per dataset and horizon.   ~5 h
  fullshot3  FEDformer and Autoformer, the same way.                            ~6 h

The hours are for one T4 and are rough. Nothing is lost if a stage is cut off part-way.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
RUNNER = os.path.join(REPO, "kaggle", "kernel_runner.py")

STAGES = ("tsfmfix", "tsfmval", "fullval", "fullshot2", "fullshot3",
          "core", "ablations", "fullshot", "tweaks", "faithful", "tsfm", "smoke")


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def restore(out_dir: str, src: str) -> int:
    """Copy a previous run's artifacts back into the checkout, so caches are reused."""
    prev = os.path.join(out_dir, "working", "outputs")
    n = 0
    for root, _, files in os.walk(prev):
        for f in files:
            p = os.path.join(root, f)
            dst = os.path.join(src, os.path.relpath(p, prev))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst) or os.path.getmtime(p) > os.path.getmtime(dst):
                shutil.copy2(p, dst)
                n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--out", required=True,
                    help="a directory that outlives the session (on Colab, a Drive folder)")
    ap.add_argument("--tsfm-models", default="ALL",
                    help="comma list, for the tsfm* stages, e.g. Moirai-S")
    ap.add_argument("--hours", type=float, default=11.0,
                    help="stop starting new commands after this long (default 11)")
    ap.add_argument("--src", default=REPO, help="the checkout to run from")
    ap.add_argument("--commit", default=None, help="commit to run (default: this checkout's HEAD)")
    ap.add_argument("--dry-run", action="store_true", help="render the runner and stop")
    args = ap.parse_args()

    commit = args.commit or git("rev-parse", "HEAD")
    url = git("remote", "get-url", "origin")
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    n = restore(out, args.src)
    print(f"[run_stage] restored {n} file(s) from {out}")

    code = open(RUNNER, encoding="utf-8").read()
    code = (code.replace("__REPO_URL__", url).replace("__COMMIT__", commit)
                .replace("__STAGE__", args.stage).replace("__TSFM_MODELS__", args.tsfm_models))
    rendered = os.path.join(out, f"kernel_runner_{args.stage}.py")
    with open(rendered, "w", encoding="utf-8") as f:
        f.write(code)

    env = os.environ.copy()
    env["EZERO_WORK"] = os.path.join(out, "working")
    env["EZERO_SRC"] = os.path.abspath(args.src)
    env["EZERO_DEADLINE_H"] = str(args.hours)
    env["PYTHONUNBUFFERED"] = "1"
    print(f"[run_stage] stage={args.stage} commit={commit[:10]} out={out}")
    if args.dry_run:
        print(f"[run_stage] dry run: wrote {rendered}")
        return 0
    return subprocess.run([sys.executable, rendered], env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
