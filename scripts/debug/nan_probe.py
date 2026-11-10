#!/usr/bin/env python3
"""
nan_probe.py - locate the first non-finite value during pre-training (GPU diagnostics).

Runs the exact pre-training setup of benchmarks/pretrain.py for a few hundred steps with
per-step checks and reports the first step / module where a forward output or a gradient
becomes non-finite. ``--mixer-eps 0`` reproduces the original group-std formulation.

Usage:
  python scripts/debug/nan_probe.py --steps 400 --mixer-eps 0
  python scripts/debug/nan_probe.py --steps 400 --mixer-eps 1e-4
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import ezero.models.components as comp  # noqa: E402
from ezero.data.lotsa import load_corpus  # noqa: E402
from ezero.data.pretrain import WindowSampler  # noqa: E402
from ezero.models.ezero import EZeroConfig, EZeroModel  # noqa: E402
from ezero.training import _to_tensors, masked_loss  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--mixer-eps", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "ezero_base.yaml")))
    corpus = yaml.safe_load(open(os.path.join(ROOT, "configs", "pretrain_corpus.yaml")))
    root = os.environ.get("EZERO_LOTSA_DIR", os.path.join(ROOT, "data", "lotsa"))
    panels = load_corpus(corpus["subsets"], root, download=True, verbose=False)
    sp = cfg["sampler"]
    sampler = WindowSampler(panels, 512, 720, sp["min_context"], 1, sp["max_group_channels"], sp["alpha"],
                            sp["val_fraction"], a.seed)

    orig = comp.group_stats
    min_var = {"v": float("inf")}

    def patched(x, group, eps=a.mixer_eps):
        mean, std, cnt = orig(x, group, eps=eps)
        multi = cnt.squeeze(1) > 1
        if multi.any():
            min_var["v"] = min(min_var["v"], float((std[multi] ** 2 - eps).min()))
        return mean, std, cnt

    comp.group_stats = patched

    torch.manual_seed(a.seed)
    model = EZeroModel(EZeroConfig.from_dict(cfg["model"])).to(dev)
    bad_fwd = []

    def hook(name):
        def f(mod, inp, out):
            if torch.is_tensor(out) and not torch.isfinite(out).all() and not bad_fwd:
                bad_fwd.append(name)
        return f

    for n, m in model.named_modules():
        if n:
            m.register_forward_hook(hook(n))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]),
                            weight_decay=float(cfg["training"]["weight_decay"]))
    report = {"device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "torch": torch.__version__,
              "mixer_eps": a.mixer_eps, "first_bad_step": None}
    model.train()
    for step in range(1, a.steps + 1):
        min_var["v"] = float("inf")
        bad_fwd.clear()
        loss = masked_loss(model, *_to_tensors(sampler.sample(64), dev))
        opt.zero_grad()
        loss.backward()
        bad_grads = [n for n, q in model.named_parameters() if q.grad is not None and not torch.isfinite(q.grad).all()]
        max_grad = max(float(q.grad.abs().max()) for q in model.parameters() if q.grad is not None
                       and torch.isfinite(q.grad).all()) if len(bad_grads) < len(list(model.parameters())) else None
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        if not torch.isfinite(loss) or bad_grads or bad_fwd or not torch.isfinite(gn):
            report.update(first_bad_step=step, loss=float(loss), grad_norm=float(gn), bad_forward_module=bad_fwd[:1],
                          bad_grad_params=bad_grads[:10], n_bad_grad_params=len(bad_grads),
                          min_group_var_this_step=min_var["v"], max_finite_abs_grad=max_grad)
            break
        opt.step()
        if step % 50 == 0:
            print(f"step {step} loss {float(loss):.4f} gradnorm {float(gn):.3f} min_group_var {min_var['v']:.3e}",
                  flush=True)
    report["steps_run"] = step
    print(json.dumps(report, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(report, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
