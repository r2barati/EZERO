#!/usr/bin/env python3
"""
pretrain.py - pre-train E-ZERO (and its ablation variants) on the LOTSA corpus.

Usage:
  python benchmarks/pretrain.py --variant base
  python benchmarks/pretrain.py --variant ALL --steps 60000          # base + 7 ablations (Table 5)
  python benchmarks/pretrain.py --variant base --synthetic --steps 200  # CPU smoke test
  python benchmarks/pretrain.py --list

Outputs:
  data/models/ezero_<variant>.pt                 checkpoint (config + weights + training summary)
  results/training_logs/ezero_<variant>.csv      step, train_loss, val_loss, lr, elapsed
  results/training_logs/ezero_<variant>.json     summary (best step, corpus mix, timings)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml  # noqa: E402

from ezero.checkpoint import save_checkpoint  # noqa: E402
from ezero.data.lotsa import load_corpus  # noqa: E402
from ezero.data.pretrain import WindowSampler, synthetic_panels  # noqa: E402
from ezero.models.ezero import ABLATION_VARIANTS, EZeroConfig, EZeroModel  # noqa: E402
from ezero.training import pretrain  # noqa: E402

MODELS_DIR = os.path.join(PROJECT_ROOT, "data", "models")
LOG_DIR = os.path.join(PROJECT_ROOT, "results", "training_logs")


def main(argv=None):
    p = argparse.ArgumentParser(description="Pre-train E-ZERO on LOTSA")
    p.add_argument("--variant", default="base", help="base | noMRC | ... | ALL | comma list")
    p.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "ezero_base.yaml"))
    p.add_argument("--corpus", default=os.path.join(PROJECT_ROOT, "configs", "pretrain_corpus.yaml"))
    p.add_argument("--lotsa-root", default=os.environ.get("EZERO_LOTSA_DIR", os.path.join(PROJECT_ROOT, "data", "lotsa")))
    p.add_argument("--out-dir", default=MODELS_DIR)
    p.add_argument("--log-dir", default=LOG_DIR, help="training curves / summaries (default results/training_logs)")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--time-budget-min", type=float, default=None, help="stop each variant after N minutes")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--synthetic", action="store_true", help="use a small synthetic corpus (tests / smoke runs)")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--list", action="store_true")
    args = p.parse_args(argv)

    if args.list:
        for k, v in ABLATION_VARIANTS.items():
            print(f"{k:10s} {v}")
        return 0

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tr, sp = cfg["training"], cfg["sampler"]
    steps = args.steps or tr["steps"]
    bs = args.batch_size or tr["batch_size"]
    seed = tr["seed"] if args.seed is None else args.seed
    variants = list(ABLATION_VARIANTS) if args.variant == "ALL" else args.variant.split(",")

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    if args.synthetic:
        panels = synthetic_panels(48, 4000, seed)
        corpus_desc = "synthetic"
    else:
        with open(args.corpus) as f:
            corpus = yaml.safe_load(f)
        panels = load_corpus(corpus["subsets"], args.lotsa_root, download=not args.no_download)
        corpus_desc = [s["name"] for s in corpus["subsets"]]
    load_s = time.time() - t0

    base_model_cfg = EZeroConfig.from_dict(cfg["model"])
    for var in variants:
        if var not in ABLATION_VARIANTS:
            raise SystemExit(f"unknown variant {var}")
        out = os.path.join(args.out_dir, f"ezero_{var}.pt")
        if args.skip_existing and os.path.exists(out):
            print(f"[pretrain] {var}: {out} exists, skipping")
            continue
        mcfg = EZeroConfig.from_dict({**base_model_cfg.to_dict(), **ABLATION_VARIANTS[var]})
        model = EZeroModel(mcfg)
        sampler = WindowSampler(panels, mcfg.context_length, mcfg.horizon, sp["min_context"], 1,
                                sp["max_group_channels"], sp["alpha"], sp["val_fraction"], seed)
        print(f"[pretrain] variant={var} params={model.num_parameters() / 1e6:.3f}M device={device} steps={steps}")
        print(f"[pretrain] dataset mix: {sampler.describe()}")
        summary = pretrain(model, sampler, steps=steps, batch_size=bs, lr=float(tr["learning_rate"]),
                           weight_decay=float(tr["weight_decay"]), warmup=int(tr["warmup_steps"]),
                           grad_clip=float(tr["gradient_clipping"]), eval_every=args.eval_every or tr["eval_every"],
                           val_batches=tr["val_batches"], patience=tr["early_stopping_patience"], device=device,
                           log_path=os.path.join(args.log_dir, f"ezero_{var}.csv"), seed=seed, amp=args.amp,
                           time_budget_s=None if args.time_budget_min is None else 60 * args.time_budget_min)
        summary.update(variant=var, params=model.num_parameters(), device=device, corpus=corpus_desc,
                       corpus_load_seconds=round(load_s, 1), dataset_mix=sampler.describe(),
                       n_panels=len(panels), n_channels=sum(p_.channels for p_ in panels),
                       synthetic=args.synthetic)
        sha = save_checkpoint(model.cpu(), out, extra={"training": summary})
        summary["checkpoint_sha256"] = sha
        os.makedirs(args.log_dir, exist_ok=True)
        with open(os.path.join(args.log_dir, f"ezero_{var}.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[pretrain] saved {out} best_val={summary['best_val_loss']:.4f} @ {summary['best_step']} "
              f"({summary['train_seconds']}s, stopped={summary['stopped']})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
