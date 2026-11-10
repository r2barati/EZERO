#!/usr/bin/env python3
"""
compare_reported.py - reproduction check: our MSE/MAE vs the numbers each model's paper reports.

Reported numbers live in results/reported/*.csv (one row per model variant, dataset, horizon, with
the paper, table, lookback and source URL). Each reported row is matched to our run of the same
model on the same dataset and horizon on the full test set with train-split scaling:

* "matched"  - our run used the paper's lookback (validation runs with --context-length);
* "L512"     - no run at the paper's lookback, compared with our canonical 512-step run
               (protocol differs; the gap is then not an implementation check).

Writes paper/tables/tableV_reproduction_rows.csv (every row) and tableV_reproduction.md (per model
and protocol: number of cells, median and max relative MSE gap, share within 5% / 10%).
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "tables")
NAME = {"ECL": "Electricity", "electricity": "Electricity", "weather": "Weather", "traffic": "Traffic"}
# (reported model, reported variant) -> our registry id; rows with other variants are not compared
VARIANT = {
    ("TTM", "TTM-Q"): "TTM-R1",            # public r1 "512-96" checkpoint (HF card)
    ("TTM", "TTM-B"): "TTM-R2",            # r2 512-context model (retrained; approximate)
    ("Moirai", "Moirai-small"): "Moirai-S",
    ("TimesFM", "TimesFM-200M"): "TimesFM-1.0",
    ("TimesFM", "TimesFM"): "TimesFM-1.0",
    ("Chronos", "Chronos-tiny"): "Chronos-T",
    ("Lag-Llama", "Lag-Llama"): "Lag-Llama",
}


def load_reported() -> pd.DataFrame:
    frames = []
    for p in sorted(glob.glob(os.path.join(RES, "reported", "*.csv"))):
        d = pd.read_csv(p)
        if "lookback" not in d.columns and "context_length" in d.columns:
            d = d.rename(columns={"context_length": "lookback"})
        d["source_file"] = os.path.basename(p)
        frames.append(d)
    if not frames:
        raise SystemExit("no reported numbers in results/reported/")
    rep = pd.concat(frames, ignore_index=True)
    rep["dataset"] = rep["dataset"].replace(NAME)
    zero_shot = rep["model"].isin({m for m, _ in VARIANT})
    rep["our_model"] = [VARIANT.get((m, v)) if z else m for m, v, z in zip(rep["model"], rep["variant"], zero_shot)]
    rep = rep[rep["our_model"].notna()]
    tbl = rep["table"].astype(str)
    rep["protocol"] = np.where(tbl.str.contains("last-window"), "last", "full")
    rep = rep[pd.to_numeric(rep["mse"], errors="coerce").notna()].copy()
    rep["mse"] = rep["mse"].astype(float)
    rep["mae"] = pd.to_numeric(rep["mae"], errors="coerce")
    rep = rep[pd.to_numeric(rep["horizon"], errors="coerce").notna()]
    rep["horizon"] = rep["horizon"].astype(int)
    return rep


def main():
    long_p = os.path.join(RES, "benchmark_long.csv")
    if not os.path.exists(long_p):
        raise SystemExit("run benchmarks/run_benchmarks.py --merge-only first")
    ours = pd.read_csv(long_p)
    if "smoke" in ours.columns:
        ours = ours[~ours["smoke"].astype(bool)]
    ours = ours[ours["protocol"].isin(["full", "last"]) & (ours["scaler"] == "train")]
    ours["context_length"] = ours["context_length"].fillna(512).astype(int)
    rep = load_reported()
    rows = []
    for _, r in rep.iterrows():
        cand = ours[(ours["model"] == r["our_model"]) & (ours["dataset"] == r["dataset"])
                    & (ours["horizon"] == r["horizon"]) & (ours["protocol"] == r["protocol"])]
        lb = pd.to_numeric(str(r.get("lookback")).rstrip("*"), errors="coerce")
        hit, how = None, None
        if not np.isnan(lb) and (cand["context_length"] == int(lb)).any():
            hit, how = cand[cand["context_length"] == int(lb)].iloc[-1], "matched"
        elif (cand["context_length"] == 512).any():
            hit, how = cand[cand["context_length"] == 512].iloc[-1], "L512"
        if hit is None:
            continue
        gap = (hit["mse"] - r["mse"]) / r["mse"]
        rows.append(dict(model=r["our_model"], variant=r.get("variant"), paper=r.get("paper"), table=r.get("table"),
                         protocol=r["protocol"],
                         dataset=r["dataset"], horizon=r["horizon"], reported_lookback=r.get("lookback"),
                         our_lookback=int(hit["context_length"]), comparison=how,
                         reported_mse=r["mse"], our_mse=round(float(hit["mse"]), 4),
                         mse_gap_pct=round(100 * gap, 1), reported_mae=r.get("mae"),
                         our_mae=round(float(hit["mae"]), 4), source_url=r.get("source_url")))
    if not rows:
        print("[compare] no overlapping rows yet")
        return
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "tableV_reproduction_rows.csv"), index=False)
    summ = []
    df["variant"] = df["variant"].astype(str) + " / " + df["paper"].astype(str) + " " + df["table"].astype(str)
    for (m, v, how), g in df.groupby(["model", "variant", "comparison"]):
        a = g["mse_gap_pct"].abs()
        summ.append(dict(model=m, variant=v, comparison=how, cells=len(g), median_gap_pct=round(g["mse_gap_pct"].median(), 1),
                         max_abs_gap_pct=round(a.max(), 1), within_5pct=f"{(a <= 5).mean():.0%}",
                         within_10pct=f"{(a <= 10).mean():.0%}"))
    s = pd.DataFrame(summ).sort_values(["comparison", "model"])
    with open(os.path.join(OUT, "tableV_reproduction.md"), "w") as f:
        f.write("**Reproduction check: our MSE vs the MSE reported in each model's paper** "
                "(gap = ours / reported - 1; positive = we are worse)\n\n")
        f.write("| model | paper variant | comparison | cells | median gap % | max abs gap % | within 5% | within 10% |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for _, r in s.iterrows():
            f.write(f"| {r['model']} | {r['variant']} | {r['comparison']} | {r['cells']} | {r['median_gap_pct']:+.1f} | "
                    f"{r['max_abs_gap_pct']:.1f} | {r['within_5pct']} | {r['within_10pct']} |\n")
        f.write("\n`matched` = run at the paper's lookback; `L512` = only our canonical 512-step run exists, so the "
                "gap mixes protocol and implementation differences. Row-level data: tableV_reproduction_rows.csv.\n")
    print(s.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
