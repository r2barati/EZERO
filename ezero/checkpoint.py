"""Checkpoint I/O. A checkpoint is a single ``.pt`` file holding the config and weights."""
from __future__ import annotations

import hashlib
import os

import torch

from ezero.models.ezero import EZeroConfig, EZeroModel


def save_checkpoint(model: EZeroModel, path: str, extra: dict | None = None) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "format": "ezero-v2",
        "config": model.config.to_dict(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "extra": extra or {},
    }
    torch.save(payload, path)
    return file_sha256(path)


def load_checkpoint(path: str, map_location="cpu") -> EZeroModel:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "ezero-v2":
        raise ValueError(f"{path} is not an E-ZERO v2 checkpoint")
    model = EZeroModel(EZeroConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["state_dict"])
    model.checkpoint_extra = payload.get("extra", {})
    model.eval()
    return model


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
