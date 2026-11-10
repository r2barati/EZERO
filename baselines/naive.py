"""Zero-shot statistical baselines (sanity floor for every table)."""
from __future__ import annotations

import numpy as np

from baselines.base import Forecaster
from ezero.frequency import season_length


class NaiveForecaster(Forecaster):
    name = "Naive"
    group = "naive"

    def predict(self, context, freq, horizon):
        return np.repeat(context[:, -1:, :], horizon, axis=1)


class MeanForecaster(Forecaster):
    name = "Mean"
    group = "naive"

    def predict(self, context, freq, horizon):
        return np.repeat(context.mean(axis=1, keepdims=True), horizon, axis=1)


class SeasonalNaiveForecaster(Forecaster):
    name = "SeasonalNaive"
    group = "naive"

    def predict(self, context, freq, horizon):
        m = max(1, min(season_length(freq), context.shape[1]))
        last = context[:, -m:, :]
        reps = int(np.ceil(horizon / m))
        return np.tile(last, (1, reps, 1))[:, :horizon, :]
