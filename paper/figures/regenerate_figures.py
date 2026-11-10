#!/usr/bin/env python3
"""
regenerate_figures.py - redraw the thesis results figures from results/*.csv.

  fig4_zero_shot_by_dataset.png    Fig. 4-6: zero-shot MSE per dataset (small multiples)
  fig7_mean_error.png              Fig. 7: mean error across datasets
  fig8_vs_architectures.png        Fig. 8: E-ZERO vs full-shot architectures per dataset
  fig9_inference_cost.png          Fig. 9: GPU time and memory (two panels, no dual axis)
  fig10_efficiency_scores.png      Fig. 10: efficiency scores
  fig11_ablation.png               Fig. 11: ablation impact (MSE increase over the base model)
  figS_training_curves.png         pre-training validation loss of every variant

E-ZERO is drawn in the highlight colour, every other model in neutral grey, so identity
never depends on telling many hues apart; values are labelled on the bars.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from extract_tables import ABL_LABEL, ABL_ORDER, ARCH_ORDER, TSFM_ORDER, avg_table, load_long  # noqa: E402

RES = os.path.join(ROOT, "results")
OUT = HERE

# reference palette (dataviz skill): surface, inks, highlight = categorical slot 1
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
HIGHLIGHT, NEUTRAL = "#2a78d6", "#b5b3ab"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "legend.frameon": False, "font.family": "sans-serif",
})


def _color(m):
    return HIGHLIGHT if m == "E-ZERO" else NEUTRAL


def _hbar(ax, labels, values, title=None, fmt="{:.3f}", log=False):
    y = np.arange(len(labels))
    ax.barh(y, values, color=[_color(m) for m in labels], height=0.62, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    if log:
        ax.set_xscale("log")
    vmax = np.nanmax(values) if len(values) else 1
    for yi, v in zip(y, values):
        if np.isfinite(v):
            ax.text(v * (1.03 if log else 1) + (0 if log else vmax * 0.015), yi, fmt.format(v), va="center",
                    fontsize=7.5, color=INK2)
    if not log:
        ax.set_xlim(0, vmax * 1.22)
    if title:
        ax.set_title(title, loc="left")


def fig_small_multiples(t: pd.DataFrame, name: str, suptitle: str):
    ds = [d for d in t.index if d != "Average"]
    if not ds:
        return
    n = len(ds)
    cols = 4 if n > 4 else n
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 0.32 * len(t.columns) * rows + 1.2 * rows),
                             squeeze=False)
    for ax, d in zip(axes.flat, ds):
        r = t.loc[d].dropna()
        _hbar(ax, list(r.index), r.values, d)
        ax.tick_params(axis="y", labelsize=7.5)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(suptitle, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=200)
    plt.close(fig)
    print(f"[figures] {name}")


def fig_mean(t: pd.DataFrame, name: str, title: str):
    if "Average" not in t.index:
        return
    r = t.loc["Average"].dropna().sort_values()
    fig, ax = plt.subplots(figsize=(6, 0.34 * len(r) + 1))
    _hbar(ax, list(r.index), r.values, title)
    ax.set_xlabel("MSE averaged over datasets and horizons (lower is better)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=200)
    plt.close(fig)
    print(f"[figures] {name}")


def fig_efficiency():
    p = os.path.join(RES, "efficiency.csv")
    if not os.path.exists(p):
        return
    eff = pd.read_csv(p)
    panels = [(c, lab, log) for c, lab, log in [("gpu_time_ms", "GPU time per batch (ms, log scale)", True),
                                                 ("gpu_mem_gb", "Peak GPU memory (GB)", False)]
              if c in eff.columns and eff[c].notna().any()]
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 0.34 * len(eff) + 1.2), squeeze=False)
        for ax, (c, lab, log) in zip(axes.flat, panels):
            r = eff[["model", c]].dropna().sort_values(c)
            _hbar(ax, list(r["model"]), r[c].values, lab, "{:.3g}", log=log)
        fig.suptitle("Inference cost: 32 series, context 512, horizon 96", x=0.01, ha="left", fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig9_inference_cost.png"), dpi=200)
        plt.close(fig)
        print("[figures] fig9_inference_cost.png")
    if "efficiency_score" in eff.columns and eff["efficiency_score"].notna().any():
        r = eff[["model", "efficiency_score"]].dropna().sort_values("efficiency_score")
        fig, ax = plt.subplots(figsize=(6, 0.34 * len(r) + 1))
        _hbar(ax, list(r["model"]), r["efficiency_score"].values, "Efficiency score (Eq. 27, lower is better)",
              "{:.4f}")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig10_efficiency_scores.png"), dpi=200)
        plt.close(fig)
        print("[figures] fig10_efficiency_scores.png")


def fig_ablation(long):
    t = avg_table(long, "full", "train", ABL_ORDER)
    if t.empty or "E-ZERO" not in t.columns or "Average" not in t.index:
        return
    base = t["E-ZERO"]
    rel = (t.drop(columns=["E-ZERO"]).div(base, axis=0) - 1.0) * 100
    r = rel.loc["Average"].dropna().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6.4, 0.38 * len(r) + 1.2))
    y = np.arange(len(r))
    ax.barh(y, r.values, color=HIGHLIGHT, height=0.62, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(y, [ABL_LABEL[m] for m in r.index])
    ax.invert_yaxis()
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.grid(axis="y", visible=False)
    span = max(1.0, np.nanmax(np.abs(r.values)))
    for yi, v in zip(y, r.values):
        ax.text(v + (span * 0.02 if v >= 0 else -span * 0.02), yi, f"{v:+.1f}%", va="center",
                ha="left" if v >= 0 else "right", fontsize=7.5, color=INK2)
    ax.set_xlim(min(0, np.nanmin(r.values)) - span * 0.25, max(0, np.nanmax(r.values)) + span * 0.25)
    ax.set_xlabel("Change in average MSE vs the full model (positive = component helps)")
    ax.set_title("Ablation: removing one component at a time", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig11_ablation.png"), dpi=200)
    plt.close(fig)
    print("[figures] fig11_ablation.png")


def fig_training_curves():
    logs = os.path.join(RES, "training_logs")
    if not os.path.isdir(logs):
        return
    series = []
    for i, m in enumerate(ABL_ORDER):
        var = "base" if m == "E-ZERO" else m.replace("E-ZERO-", "")
        p = os.path.join(logs, f"ezero_{var}.csv")
        if os.path.exists(p):
            series.append((ABL_LABEL[m], pd.read_csv(p), CATEGORICAL[i % len(CATEGORICAL)]))
    if not series:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for lab, df, c in series:
        ax.plot(df["step"], df["val_loss"], color=c, linewidth=2, label=lab)
    ax.set_xlabel("pre-training step")
    ax.set_ylabel("validation MSE (instance-normalised)")
    ax.set_title("Pre-training on LOTSA: validation loss", loc="left")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figS_training_curves.png"), dpi=200)
    plt.close(fig)
    print("[figures] figS_training_curves.png")


def main():
    long = load_long()
    sub = sorted(p for p in long["protocol"].unique() if p.startswith("sub"))
    sub = sub[0] if sub else "sub16"
    t2 = avg_table(long, sub, "train", TSFM_ORDER)
    fig_small_multiples(t2, "fig4_zero_shot_by_dataset.png",
                        f"Zero-shot MSE by dataset (avg over horizons, protocol {sub})")
    fig_mean(t2, "fig7_mean_error.png", "Mean zero-shot error across datasets")
    t3 = avg_table(long, "full", "train", ARCH_ORDER)
    fig_small_multiples(t3, "fig8_vs_architectures.png",
                        "Zero-shot E-ZERO vs full-shot architectures (MSE, full test set)")
    fig_efficiency()
    fig_ablation(long)
    fig_training_curves()


if __name__ == "__main__":
    main()
