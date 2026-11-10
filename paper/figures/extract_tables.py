#!/usr/bin/env python3
"""
extract_tables.py - regenerate the thesis results tables from results/*.csv.

Writes Markdown and LaTeX versions to paper/tables/:

  table2_zero_shot_vs_tsfm      Table 2: zero-shot E-ZERO vs thesis-era TS pre-trained models (sub16)
  table2c_post_thesis_models    the same windows vs foundation models released after the thesis
  table3_vs_architectures       Table 3 / Fig. 8: zero-shot E-ZERO vs full-shot architectures (full protocol)
  table3b_per_horizon           per-horizon MSE/MAE for every model (full protocol)
  table_inference_cost          Table 3 (inference time and memory)
  table4_efficiency             Table 4 (efficiency scores)
  table5_ablation               Table 5 (ablation study, full protocol)
  tableS1_normalisation         leakage-free vs legacy full-series normalisation (diagnostic)

Best and second-best entries per row are marked **bold** / _underline_ (Markdown) and
\\textbf / \\underline (LaTeX), as in the thesis. Missing cells mean the model was not run
(see results/manifest.json and results/failures.json).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from ezero.data.ltsf import DEFAULT_DATASETS, DEFAULT_HORIZONS  # noqa: E402

RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "tables")
# Table 2 mirrors the thesis columns with checkpoints that were public before the thesis
TSFM_ORDER = ["E-ZERO", "TTM-R1", "Chronos-T", "Lag-Llama", "Moirai-S", "TimesFM-1.0", "SeasonalNaive", "Naive"]
# later releases, reported separately as context
POST_THESIS_ORDER = ["E-ZERO", "TTM-R2", "Chronos-Bolt-T", "TimesFM-2.5", "SeasonalNaive"]
ARCH_ORDER = ["E-ZERO", "PatchTST", "iTransformer", "DLinear", "SeasonalNaive", "Naive", "Mean"]
ABL_ORDER = ["E-ZERO", "E-ZERO-noMRC", "E-ZERO-noDTP", "E-ZERO-noRAE", "E-ZERO-noCRRB", "E-ZERO-noExog",
             "E-ZERO-noTCN", "E-ZERO-noNBEATS"]
ABL_LABEL = {"E-ZERO": "Base (all)", "E-ZERO-noMRC": "w/o MRC", "E-ZERO-noDTP": "w/o DTP",
             "E-ZERO-noRAE": "w/o RAE", "E-ZERO-noCRRB": "w/o CRRB", "E-ZERO-noExog": "w/o Exog",
             "E-ZERO-noTCN": "w/o TCN", "E-ZERO-noNBEATS": "w/o N-BEATS"}


def load_long() -> pd.DataFrame:
    p = os.path.join(RES, "benchmark_long.csv")
    if not os.path.exists(p):
        raise SystemExit("results/benchmark_long.csv missing - run benchmarks/run_benchmarks.py --merge-only")
    df = pd.read_csv(p)
    if "smoke" in df.columns:
        df = df[~df["smoke"].astype(bool)]
    return df


def avg_table(long, protocol, scaler, models, metric="mse", context_length=512):
    """dataset x model table of the metric averaged over the four canonical horizons."""
    long = long[long["context_length"].fillna(512).astype(int) == context_length]
    d = long[(long["protocol"] == protocol) & (long["scaler"] == scaler)
             & long["horizon"].isin(DEFAULT_HORIZONS) & long["model"].isin(models)]
    cnt = d.groupby(["dataset", "model"])["horizon"].nunique()
    full = cnt[cnt == len(DEFAULT_HORIZONS)].index
    d = d.set_index(["dataset", "model"]).loc[lambda x: x.index.isin(full)].reset_index()
    t = d.groupby(["dataset", "model"])[metric].mean().unstack("model")
    t = t.reindex([x for x in DEFAULT_DATASETS if x in t.index])
    t = t[[m for m in models if m in t.columns]]
    if len(t):
        t.loc["Average"] = t.mean(axis=0, skipna=False)
    return t


def _fmt(v, nd=3):
    return "-" if pd.isna(v) else f"{v:.{nd}f}"


def to_markdown(t: pd.DataFrame, nd=3, rank=True, index_name="Dataset") -> str:
    cols = list(t.columns)
    lines = ["| " + " | ".join([index_name] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in t.iterrows():
        vals = row.values.astype(float)
        order = np.argsort(np.where(np.isnan(vals), np.inf, vals))
        cells = []
        for j, v in enumerate(vals):
            s = _fmt(v, nd)
            if rank and not np.isnan(v):
                if j == order[0]:
                    s = f"**{s}**"
                elif len(order) > 1 and j == order[1]:
                    s = f"_{s}_"
            cells.append(s)
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def to_latex(t: pd.DataFrame, caption: str, label: str, nd=3, rank=True, index_name="Dataset") -> str:
    cols = list(t.columns)
    head = " & ".join([index_name] + [c.replace("_", r"\_") for c in cols]) + r" \\"
    body = []
    for idx, row in t.iterrows():
        vals = row.values.astype(float)
        order = np.argsort(np.where(np.isnan(vals), np.inf, vals))
        cells = []
        for j, v in enumerate(vals):
            s = _fmt(v, nd)
            if rank and not np.isnan(v):
                if j == order[0]:
                    s = rf"\textbf{{{s}}}"
                elif len(order) > 1 and j == order[1]:
                    s = rf"\underline{{{s}}}"
            cells.append(s)
        body.append(" & ".join([str(idx)] + cells) + r" \\")
    return "\n".join([r"\begin{table}[ht]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
                      r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
                      head, r"\midrule", *body, r"\bottomrule", r"\end{tabular}}", r"\end{table}"])


def write(name, t, caption, nd=3, rank=True, note="", index_name="Dataset"):
    os.makedirs(OUT, exist_ok=True)
    if t is None or t.empty:
        print(f"[tables] {name}: no data yet")
        return
    with open(os.path.join(OUT, f"{name}.md"), "w") as f:
        f.write(f"**{caption}**\n\n{to_markdown(t, nd, rank, index_name)}\n")
        if note:
            f.write(f"\n{note}\n")
    with open(os.path.join(OUT, f"{name}.tex"), "w") as f:
        f.write(to_latex(t, caption, f"tab:{name}", nd, rank, index_name) + "\n")
    t.to_csv(os.path.join(OUT, f"{name}.csv"))
    print(f"[tables] {name}: {t.shape[0]} rows x {t.shape[1]} models")


def per_horizon(long, protocol, scaler, models, context_length=512):
    long = long[long["context_length"].fillna(512).astype(int) == context_length]
    d = long[(long["protocol"] == protocol) & (long["scaler"] == scaler) & long["model"].isin(models)]
    if d.empty:
        return None
    t = d.pivot_table(index=["dataset", "horizon"], columns="model", values="mse")
    t = t[[m for m in models if m in t.columns]]
    t.index = [f"{a} / {b}" for a, b in t.index]
    order = {f"{ds} / {h}": (i, h) for i, ds in enumerate(DEFAULT_DATASETS) for h in DEFAULT_HORIZONS}
    return t.loc[sorted(t.index, key=lambda k: order.get(k, (99, 0)))]


def main():
    long = load_long()
    sub = sorted(p for p in long["protocol"].unique() if p.startswith("sub"))
    sub = sub[0] if sub else "sub16"
    write("table2_zero_shot_vs_tsfm", avg_table(long, sub, "train", TSFM_ORDER),
          f"Zero-shot MSE vs TS pre-trained models available at the time of the thesis, averaged over horizons {{96, 192, 336, 720}} "
          f"(protocol {sub}: identical test windows for every model, train-split scaling)")
    write("table2b_zero_shot_vs_tsfm_mae", avg_table(long, sub, "train", TSFM_ORDER, "mae"),
          f"Zero-shot MAE vs TS pre-trained models (protocol {sub})")
    write("table2c_post_thesis_models", avg_table(long, sub, "train", POST_THESIS_ORDER),
          f"Zero-shot MSE vs foundation models released after the thesis (protocol {sub}); context only")
    write("table3_vs_architectures", avg_table(long, "full", "train", ARCH_ORDER),
          "Long-sequence forecasting MSE, full test set (stride 1): zero-shot E-ZERO vs full-shot "
          "architectures trained on each dataset")
    write("table3b_per_horizon", per_horizon(long, "full", "train", ARCH_ORDER + ABL_ORDER[1:]),
          "Per-horizon MSE, full protocol", index_name="Dataset / horizon")
    abl = avg_table(long, "full", "train", ABL_ORDER)
    if not abl.empty:
        abl = abl.rename(columns=ABL_LABEL)
    write("table5_ablation", abl, "E-ZERO ablations: MSE with and without each component (full protocol)",
          rank=False)
    sc = []
    for s in ("train", "full"):
        t = avg_table(long, "full", s, ["E-ZERO"])
        if not t.empty:
            sc.append(t.rename(columns={"E-ZERO": f"E-ZERO (scaler={s})"}))
    if len(sc) == 2:
        write("tableS1_normalisation", pd.concat(sc, axis=1),
              "E-ZERO MSE with leakage-free train-split scaling vs the thesis' full-series scaling",
              rank=False, note="The thesis (Sec. 3.6.1) normalised with statistics of the whole series and "
                               "flagged it as possible leakage; the repository default is train-split scaling.")

    eff_p = os.path.join(RES, "efficiency.csv")
    if os.path.exists(eff_p):
        eff = pd.read_csv(eff_p).set_index("model")
        cols = [c for c in ["gpu_time_ms", "params_m", "gpu_mem_gb", "cpu_time_s"] if c in eff.columns]
        cost = eff[cols].rename(columns={"gpu_time_ms": "GPU TIME (ms)", "params_m": "Params (M)",
                                         "gpu_mem_gb": "MEM (GB)", "cpu_time_s": "CPU TIME (s)"})
        write("table_inference_cost", cost, "Inference time per batch (32 series, context 512, horizon 96) "
              "and memory", nd=4, rank=False, index_name="Model")
        if "efficiency_score" in eff.columns:
            write("table4_efficiency", eff[["efficiency_score"]].sort_values("efficiency_score")
                  .rename(columns={"efficiency_score": "Efficiency score"}),
                  "Efficiency score (Eq. 27, equal weights; lower is better)", nd=4, rank=False, index_name="Model")


if __name__ == "__main__":
    main()
