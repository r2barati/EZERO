"""Wrap an E-ZERO checkpoint as a forecaster (one checkpoint covers all horizons <= 720)."""
from __future__ import annotations

import numpy as np
import torch

from baselines.base import Forecaster, from_series, to_series
from ezero.checkpoint import file_sha256, load_checkpoint
from ezero.frequency import bucket_id


class EZeroForecaster(Forecaster):
    group = "ezero"

    def __init__(self, checkpoint: str | None = None, model=None, name: str = "E-ZERO", device: str | None = None,
                 batch_series: int = 4096, use_groups: bool = True):
        self.name = name
        self.checkpoint = checkpoint
        self.model = model if model is not None else load_checkpoint(checkpoint)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.batch_series = batch_series
        self.use_groups = use_groups
        self.sha256 = file_sha256(checkpoint) if checkpoint else None

    def to_device(self, device):
        self.device = device
        self.model.to(device)
        return self

    def torch_module(self):
        return self.model

    @torch.no_grad()
    def predict(self, context, freq, horizon):
        W, L, C = context.shape
        series = torch.from_numpy(to_series(context)).to(self.device)
        group = torch.arange(W, device=self.device).repeat_interleave(C) if self.use_groups else None
        fb = bucket_id(freq)
        outs = []
        # keep whole windows together so the mixer sees complete channel groups
        step = max(C, (self.batch_series // C) * C)
        for k in range(0, series.shape[0], step):
            g = None if group is None else group[k:k + step] - group[k]
            outs.append(self.model(series[k:k + step], fb, group=g, horizon=horizon).float().cpu())
        pred = torch.cat(outs).numpy()
        return from_series(pred, W, C)

    def info(self):
        return {"checkpoint": self.checkpoint, "checkpoint_sha256": self.sha256,
                "params": self.model.num_parameters(), "config": self.model.config.to_dict()}
