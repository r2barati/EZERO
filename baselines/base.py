"""Common forecaster interface.

``predict(context (W, L, C), freq, horizon) -> (W, horizon, C)`` on standard-scaled values.
"""
from __future__ import annotations

import numpy as np


class Forecaster:
    name = "base"
    group = "other"          # ezero | ezero-ablation | naive | tsfm | fullshot
    zero_shot = True
    multi_horizon = True     # prefix of the longest forecast is valid for shorter horizons
    gpu_only = False

    def fit(self, data, horizon: int):  # full-shot models override
        return self

    def predict(self, context: np.ndarray, freq: str, horizon: int) -> np.ndarray:
        raise NotImplementedError

    def to_device(self, device: str):
        return self

    def torch_module(self):
        return None

    def info(self) -> dict:
        return {}


def to_series(context: np.ndarray) -> np.ndarray:
    """(W, L, C) -> (W*C, L) with channel index fastest."""
    W, L, C = context.shape
    return np.ascontiguousarray(context.transpose(0, 2, 1).reshape(W * C, L))


def from_series(pred: np.ndarray, W: int, C: int) -> np.ndarray:
    """(W*C, H) -> (W, H, C)."""
    H = pred.shape[-1]
    return pred.reshape(W, C, H).transpose(0, 2, 1)
