"""Zero-shot time-series foundation-model baselines (thesis Sec. 4.1, "TS pre-trained models").

Each wrapper imports its package lazily, so the core repository has no hard dependency on
them; ``kaggle/kernels/ezero-tsfm-*`` installs one package per kernel to avoid version
conflicts between them. All are evaluated channel-independently on standard-scaled inputs
with a 512-step context, and the point forecast is the predictive mean.

| id               | checkpoint                                   | package               |
|------------------|----------------------------------------------|-----------------------|
| Chronos-T        | amazon/chronos-t5-tiny (8M)                  | chronos-forecasting   |
| Chronos-Bolt-T   | amazon/chronos-bolt-tiny (9M)                | chronos-forecasting   |
| TTM              | ibm-granite/granite-timeseries-ttm-r2        | granite-tsfm          |
| Moirai-S         | Salesforce/moirai-1.0-R-small (14M)          | uni2ts                |
| TimesFM          | google/timesfm-1.0-200m-pytorch (200M)       | timesfm (1.x API)     |
"""
from __future__ import annotations

import numpy as np

from baselines.base import Forecaster, from_series, to_series
from ezero.frequency import bucket_name


def _device(device=None):
    import torch
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


class _SeriesForecaster(Forecaster):
    """Helper: subclasses implement ``_predict_series((N, L) float32, freq, H) -> (N, H)``."""
    group = "tsfm"
    batch = 256

    def predict(self, context, freq, horizon):
        W, L, C = context.shape
        series = to_series(context)
        outs = [self._predict_series(series[k:k + self.batch], freq, horizon)
                for k in range(0, len(series), self.batch)]
        return from_series(np.concatenate(outs).astype(np.float32), W, C)


class ChronosForecaster(_SeriesForecaster):
    def __init__(self, model_id="amazon/chronos-t5-tiny", name="Chronos-T", num_samples=20, device=None, batch=256):
        import torch
        from chronos import BaseChronosPipeline
        self.name = name
        self.model_id = model_id
        self.device = _device(device)
        self.num_samples = num_samples
        self.batch = batch
        self.pipe = BaseChronosPipeline.from_pretrained(model_id, device_map=self.device, torch_dtype=torch.float32)

    def torch_module(self):
        return getattr(self.pipe, "model", None)

    def to_device(self, device):
        self.device = device
        self.pipe.model.to(device)
        return self

    def _predict_series(self, x, freq, H):
        import torch
        ctx = torch.from_numpy(x)
        kw = {"quantile_levels": [0.5]}
        if "bolt" not in self.model_id:
            kw.update(num_samples=self.num_samples, limit_prediction_length=False)
        try:
            _, mean = self.pipe.predict_quantiles(ctx, prediction_length=H, **kw)
        except TypeError:
            kw.pop("limit_prediction_length", None)
            _, mean = self.pipe.predict_quantiles(ctx, prediction_length=H, **kw)
        return mean.float().cpu().numpy()

    def info(self):
        return {"checkpoint": self.model_id, "num_samples": self.num_samples}


class TTMForecaster(Forecaster):
    """TinyTimeMixer; ``get_model`` picks the checkpoint matching (context, horizon)."""
    group = "tsfm"
    multi_horizon = False

    def __init__(self, model_id="ibm-granite/granite-timeseries-ttm-r2", name="TTM", context_length=512,
                 device=None, batch=64):
        self.name = name
        self.model_id = model_id
        self.L = context_length
        self.device = _device(device)
        self.batch = batch
        self.models = {}
        self.revisions = {}

    def _model(self, H):
        if H not in self.models:
            from tsfm_public.toolkit.get_model import get_model
            m = get_model(self.model_id, context_length=self.L, prediction_length=H)
            self.models[H] = m.to(self.device).eval()
            self.revisions[H] = getattr(m.config, "_name_or_path", self.model_id)
        return self.models[H]

    def torch_module(self):
        return self._model(96)

    def to_device(self, device):
        self.device = device
        for m in self.models.values():
            m.to(device)
        return self

    def predict(self, context, freq, horizon):
        import torch
        m = self._model(horizon)
        outs = []
        with torch.no_grad():
            for k in range(0, len(context), self.batch):
                x = torch.from_numpy(context[k:k + self.batch]).to(self.device)
                out = m(past_values=x).prediction_outputs
                outs.append(out[:, :horizon].float().cpu().numpy())
        return np.concatenate(outs)

    def info(self):
        return {"checkpoint": self.model_id, "revisions": {str(k): v for k, v in self.revisions.items()}}


class MoiraiForecaster(_SeriesForecaster):
    def __init__(self, model_id="Salesforce/moirai-1.0-R-small", name="Moirai-S", context_length=512,
                 patch_size=32, num_samples=20, device=None, batch=128):
        from uni2ts.model.moirai import MoiraiModule
        self.name = name
        self.model_id = model_id
        self.L = context_length
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.device = _device(device)
        self.batch = batch
        self.module = MoiraiModule.from_pretrained(model_id)
        self.forecasters = {}

    def _fc(self, H):
        if H not in self.forecasters:
            from uni2ts.model.moirai import MoiraiForecast
            self.forecasters[H] = MoiraiForecast(
                module=self.module, prediction_length=H, context_length=self.L, patch_size=self.patch_size,
                num_samples=self.num_samples, target_dim=1, feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0).to(self.device).eval()
        return self.forecasters[H]

    def torch_module(self):
        return self.module

    def to_device(self, device):
        self.device = device
        for f in self.forecasters.values():
            f.to(device)
        return self

    def _predict_series(self, x, freq, H):
        import torch
        fc = self._fc(H)
        t = torch.from_numpy(x).to(self.device).unsqueeze(-1)            # (N, L, 1)
        with torch.no_grad():
            samples = fc(past_target=t, past_observed_target=torch.ones_like(t, dtype=torch.bool),
                         past_is_pad=torch.zeros(t.shape[:2], dtype=torch.bool, device=self.device))
        s = samples.float().cpu().numpy()
        s = s.reshape(s.shape[0], s.shape[1], -1)[:, :, :H] if s.ndim > 2 else s[:, None, :H]
        return s.mean(axis=1)

    def info(self):
        return {"checkpoint": self.model_id, "patch_size": self.patch_size, "num_samples": self.num_samples}


class TimesFMForecaster(_SeriesForecaster):
    def __init__(self, model_id="google/timesfm-1.0-200m-pytorch", name="TimesFM", context_length=512,
                 horizon_len=720, device=None, batch=256):
        import timesfm
        self.name = name
        self.model_id = model_id
        self.batch = batch
        backend = "gpu" if _device(device).startswith("cuda") else "cpu"
        self.tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(backend=backend, per_core_batch_size=32, horizon_len=horizon_len,
                                           context_len=context_length),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=model_id))
        self.gpu_only = False

    def torch_module(self):
        return getattr(self.tfm, "_model", None)

    def _predict_series(self, x, freq, H):
        b = bucket_name(freq)
        f = 0 if b in ("second", "minute", "10min", "15min", "30min", "hour", "day") else (1 if b in ("week", "month") else 2)
        point, _ = self.tfm.forecast(list(x), freq=[f] * len(x))
        return np.asarray(point)[:, :H]

    def info(self):
        return {"checkpoint": self.model_id}


TSFM_FACTORIES = {
    "Chronos-T": lambda **kw: ChronosForecaster("amazon/chronos-t5-tiny", "Chronos-T", **kw),
    "Chronos-Bolt-T": lambda **kw: ChronosForecaster("amazon/chronos-bolt-tiny", "Chronos-Bolt-T", **kw),
    "TTM": lambda **kw: TTMForecaster(**kw),
    "Moirai-S": lambda **kw: MoiraiForecaster(**kw),
    "TimesFM": lambda **kw: TimesFMForecaster(**kw),
}
