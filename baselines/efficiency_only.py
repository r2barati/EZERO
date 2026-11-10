"""Models that appear only in the thesis' inference-cost tables (Table 3, Table 4).

GPT4TS (One Fits All, Zhou et al., 2023) and MOMENT (Goswami et al., 2024) are not zero-shot
forecasters: both need a forecasting head trained on the target dataset. The thesis reports them
only for parameters, latency and memory, so they are registered as ``efficiency-only``: the
benchmark runner refuses to score their accuracy, and ``run_efficiency.py`` times one batch with
an untrained head (the cost of the forward pass does not depend on the head's weights).
"""
from __future__ import annotations

import numpy as np

from baselines.base import Forecaster, from_series, to_series


def _device(device=None):
    import torch
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


class GPT4TSForecaster(Forecaster):
    """GPT4TS long-term forecasting model: instance norm, patching (16/8), GPT-2 (first 6 layers,
    pre-trained ``gpt2`` weights, inputs_embeds) and a linear head, channel-independent."""
    group = "efficiency"
    zero_shot = False

    def __init__(self, name="GPT4TS", context_length=512, horizon=96, gpt_layers=6, patch=16, stride=8,
                 device=None):
        import torch
        import torch.nn as nn
        from transformers import GPT2Model
        self.name = name
        self.device = _device(device)
        self.patch, self.stride = patch, stride
        n_patch = (context_length - patch) // stride + 2
        gpt = GPT2Model.from_pretrained("gpt2")
        gpt.h = gpt.h[:gpt_layers]
        d = gpt.config.n_embd

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(patch, d)
                self.gpt = gpt
                self.head = nn.Linear(d * n_patch, horizon)

        self.net = Net().to(self.device).eval()
        del torch

    def torch_module(self):
        return self.net

    def to_device(self, device):
        self.device = device
        self.net.to(device)
        return self

    def predict(self, context, freq, horizon):
        import torch
        W, L, C = context.shape
        x = torch.from_numpy(to_series(context)).to(self.device)
        with torch.no_grad():
            mu = x.mean(1, keepdim=True)
            sd = x.std(1, keepdim=True) + 1e-5
            z = (x - mu) / sd
            z = torch.cat([z, z[:, -1:].expand(-1, self.stride)], dim=1).unfold(1, self.patch, self.stride)
            h = self.net.gpt(inputs_embeds=self.net.embed(z)).last_hidden_state
            y = self.net.head(h.flatten(1))[:, :horizon] * sd + mu
        return from_series(y.float().cpu().numpy(), W, C)


class MomentForecaster(Forecaster):
    """MOMENT-1-large with its forecasting head (momentfm; Python 3.11 environment)."""
    group = "efficiency"
    zero_shot = False

    def __init__(self, model_id="AutonLab/MOMENT-1-large", name="MOMENT", horizon=96, device=None, batch=64):
        from momentfm import MOMENTPipeline
        self.name = name
        self.model_id = model_id
        self.device = _device(device)
        self.batch = batch
        self.model = MOMENTPipeline.from_pretrained(
            model_id, model_kwargs={"task_name": "forecasting", "forecast_horizon": horizon, "head_dropout": 0.1,
                                    "freeze_encoder": True, "freeze_embedder": True, "freeze_head": False})
        self.model.init()
        self.model.to(self.device).eval()

    def torch_module(self):
        return self.model

    def to_device(self, device):
        self.device = device
        self.model.to(device)
        return self

    def predict(self, context, freq, horizon):
        import torch
        W, L, C = context.shape
        x = torch.from_numpy(to_series(context)).to(self.device).unsqueeze(1)   # (N, 1, 512)
        outs = []
        with torch.no_grad():
            for k in range(0, len(x), self.batch):
                outs.append(self.model(x_enc=x[k:k + self.batch]).forecast[:, 0, :horizon].float().cpu().numpy())
        return from_series(np.concatenate(outs), W, C)


EFFICIENCY_ONLY_FACTORIES = {
    "GPT4TS": lambda **kw: GPT4TSForecaster(**kw),
    "MOMENT": lambda **kw: MomentForecaster(**kw),
}
