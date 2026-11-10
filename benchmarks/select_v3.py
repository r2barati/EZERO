#!/usr/bin/env python3
"""
select_v3.py - pre-registered selection of the E-ZERO v3 configuration.

Rule (fixed before any evaluation-set result of the candidates exists):
  among the rows of results/dev/dev_results.csv with --label (default "v3-candidates"),
  choose the (checkpoint, test-time averaging) pair with the lowest development score
  (geometric mean over the held-out LOTSA datasets of MSE / MSE(seasonal naive));
  ties (< 0.5% apart) go to the base checkpoint, then to tta=none.

The chosen checkpoint is copied to data/models/ezero_v3.pt and the choice (with every candidate's
score) is written to results/dev/selection.json, which run_benchmarks.py reads for "E-ZERO-v3".
"""
import argparse
import json
import os
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd  # noqa: E402

DEV = os.path.join(ROOT, "results", "dev", "dev_results.csv")
SEL = os.path.join(ROOT, "results", "dev", "selection.json")
MODELS = os.path.join(ROOT, "data", "models")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="v3-candidates")
    a = p.parse_args(argv)
    df = pd.read_csv(DEV)
    df = df[df["label"] == a.label]
    if df.empty:
        raise SystemExit(f"no dev rows with label {a.label!r}")
    scores = df.groupby("candidate")["score"].first().sort_values()
    best = scores.iloc[0]
    near = scores[scores <= best * 1.005].index.tolist()

    def key(c):
        ck, tta = c.split("|tta=")
        return (0 if ck == "ezero_base.pt" else 1, 0 if tta == "none" else 1, scores[c])

    choice = sorted(near, key=key)[0]
    ck, tta = choice.split("|tta=")
    shutil.copy2(os.path.join(MODELS, ck), os.path.join(MODELS, "ezero_v3.pt"))
    sel = dict(rule="lowest development score; ties (<0.5%) -> base checkpoint, then tta=none",
               label=a.label, chosen=choice, checkpoint=ck, tta=tta, score=float(scores[choice]),
               all_scores={k: float(v) for k, v in scores.items()})
    os.makedirs(os.path.dirname(SEL), exist_ok=True)
    with open(SEL, "w") as f:
        json.dump(sel, f, indent=2)
    print(json.dumps(sel, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
