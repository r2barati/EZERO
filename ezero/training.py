"""Pre-training loop (thesis Sec. 3.6).

AdamW (lr 3e-4, weight decay 1e-3), batch 64, linear warm-up (10k steps) followed by
cosine annealing, gradient clipping at 3.0, dropout 0.2 (in the model), MSE loss, early
stopping on a held-out validation split. The loss is computed in the instance-normalised
space using only observed values, so no statistics of the targets leak into the inputs.
"""
from __future__ import annotations

import csv
import math
import os
import time

import numpy as np
import torch

from ezero.data.pretrain import WindowSampler
from ezero.models.ezero import EZeroModel


def _to_tensors(batch: dict, device):
    ctx = torch.from_numpy(batch["context"]).to(device)
    tgt = torch.from_numpy(batch["target"]).to(device)
    freq = torch.from_numpy(batch["freq"]).to(device)
    group = torch.from_numpy(batch["group"]).to(device)
    return ctx, tgt, freq, group


def masked_loss(model: EZeroModel, ctx, tgt, freq, group):
    cfg = model.config
    obs = torch.isfinite(ctx)
    x, mu, sigma = model.normalize(torch.nan_to_num(ctx, nan=0.0), obs)
    tmask = torch.isfinite(tgt)
    y = ((torch.nan_to_num(tgt, nan=0.0) - mu) / sigma).clamp(-cfg.target_clip, cfg.target_clip)
    pred = model.forward_normalized(x, obs, freq, group)
    w = tmask.to(pred.dtype)
    return ((pred - y) ** 2 * w).sum() / w.sum().clamp(min=1.0)


def lr_lambda(warmup: int, total: int, min_ratio: float = 0.01):
    def f(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        p = (step - warmup) / max(1, total - warmup)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return f


def pretrain(model: EZeroModel, sampler: WindowSampler, steps: int = 50000, batch_size: int = 64,
             lr: float = 3e-4, weight_decay: float = 1e-3, warmup: int = 10000, grad_clip: float = 3.0,
             eval_every: int = 1000, val_batches: int = 32, patience: int = 10, device: str = "cpu",
             log_path: str | None = None, seed: int = 0, amp: bool = False, verbose: bool = True,
             time_budget_s: float | None = None) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(device)
    warmup = min(warmup, max(1, steps // 5))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(warmup, steps))
    use_amp = amp and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # fixed validation batches (future targets of every panel)
    state = sampler.rng.bit_generator.state
    sampler.rng = np.random.default_rng(seed + 1)
    val = [sampler.sample(batch_size, split="val") for _ in range(val_batches)]
    sampler.rng = np.random.default_rng(seed)
    del state

    def val_loss():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for b in val:
                tot += float(masked_loss(model, *_to_tensors(b, device)))
                n += 1
        model.train()
        return tot / max(1, n)

    log_f = None
    writer = None
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        log_f = open(log_path, "w", newline="")
        writer = csv.writer(log_f)
        writer.writerow(["step", "train_loss", "val_loss", "lr", "elapsed_s"])

    best = float("inf")
    best_state = None
    best_step = 0
    bad = 0
    run = 0.0
    run_n = 0
    t0 = time.time()
    model.train()
    step = 0
    stopped = "max_steps"
    for step in range(1, steps + 1):
        ctx, tgt, freq, group = _to_tensors(sampler.sample(batch_size), device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss = masked_loss(model, ctx, tgt, freq, group)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if torch.isfinite(loss):
            run += float(loss.detach())
            run_n += 1
        if step % eval_every == 0 or step == steps:
            vl = val_loss()
            tl = run / max(1, run_n)
            run, run_n = 0.0, 0
            el = time.time() - t0
            if writer:
                writer.writerow([step, f"{tl:.6f}", f"{vl:.6f}", f"{sched.get_last_lr()[0]:.3e}", f"{el:.1f}"])
                log_f.flush()
            if verbose:
                print(f"[pretrain] step {step:6d} train {tl:.4f} val {vl:.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{el:.0f}s", flush=True)
            if vl < best - 1e-5:
                best, best_step, bad = vl, step, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience and step > warmup:
                    stopped = "early_stopping"
                    break
            if time_budget_s is not None and el > time_budget_s:
                stopped = "time_budget"
                break
    if log_f:
        log_f.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return dict(best_val_loss=best, best_step=best_step, last_step=step, stopped=stopped,
                train_seconds=round(time.time() - t0, 1), warmup=warmup, steps=steps, batch_size=batch_size,
                lr=lr, weight_decay=weight_decay, grad_clip=grad_clip)
