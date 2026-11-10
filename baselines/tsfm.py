"""Zero-shot time-series foundation-model baselines (thesis Sec. 4.1, "TS pre-trained models").

Each wrapper imports its package lazily, so the core repository has no hard dependency on
them; ``kaggle/kernels/ezero-tsfm-*`` installs one package per kernel to avoid version
conflicts between them. All are evaluated channel-independently on standard-scaled inputs
with a 512-step context, and the point forecast is the predictive mean.

Thesis-era checkpoints (public before the thesis, Aug/Sep 2024) are the primary comparison;
later releases are reported separately as "post-thesis" context (HuggingFace creation dates).

| id               | checkpoint (HF created)                                  | package               | era         |
|------------------|----------------------------------------------------------|-----------------------|-------------|
| Chronos-T        | amazon/chronos-t5-tiny, 8M (2024-02)                     | chronos-forecasting   | thesis      |
| TTM-R1           | ibm-granite/granite-timeseries-ttm-r1 512-96 (2024-04)   | granite-tsfm          | thesis      |
| Moirai-S         | Salesforce/moirai-1.0-R-small, 14M (2024-02)             | uni2ts                | thesis      |
| TimesFM-1.0      | google/timesfm-1.0-200m-pytorch (2024-08)                | timesfm 1.x (py3.11)  | thesis      |
| Lag-Llama        | time-series-foundation-models/Lag-Llama (2024-02)        | lag-llama (py3.11)    | thesis      |
| Chronos-Bolt-T   | amazon/chronos-bolt-tiny, 9M (2024-11)                   | chronos-forecasting   | post-thesis |
| TTM-R2           | ibm-granite/granite-timeseries-ttm-r2 (2024-10)          | granite-tsfm          | post-thesis |
| TimesFM-2.5      | google/timesfm-2.5-200m-pytorch (2025-09)                | timesfm >= 2          | post-thesis |
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
    """TimesFM-2.5-200M (timesfm >= 2). The 1.0-200m checkpoint used in the thesis is only
    loadable by timesfm 1.x, which has no build for Python 3.13 (the current Kaggle image)."""

    def __init__(self, model_id="google/timesfm-2.5-200m-pytorch", name="TimesFM-2.5", context_length=512,
                 max_horizon=768, device=None, batch=128):
        import timesfm
        import torch
        self.name = name
        self.model_id = model_id
        self.batch = batch
        self.device = _device(device)
        self.tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id, torch_compile=False)
        self.tfm.compile(timesfm.ForecastConfig(
            max_context=context_length, max_horizon=max_horizon, normalize_inputs=True,
            per_core_batch_size=batch, use_continuous_quantile_head=True, force_flip_invariance=True,
            infer_is_positive=False, fix_quantile_crossing=True))
        del torch

    def torch_module(self):
        return getattr(self.tfm, "model", None)

    def _predict_series(self, x, freq, H):
        point, _ = self.tfm.forecast(horizon=H, inputs=[r for r in x])
        return np.asarray(point)[:, :H]

    def info(self):
        return {"checkpoint": self.model_id}


class TTMR1Forecaster(Forecaster):
    """TTM-R1 (thesis era). The released R1 checkpoint forecasts 96 steps from 512; longer horizons
    are produced by rolling the model forward in 96-step chunks on its own forecasts."""
    group = "tsfm"
    multi_horizon = True

    def __init__(self, model_id="ibm-granite/granite-timeseries-ttm-r1", revision="main", name="TTM-R1",
                 context_length=512, device=None, batch=64):
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
        self.name = name
        self.model_id, self.revision = model_id, revision
        self.L = context_length
        self.device = _device(device)
        self.batch = batch
        self.model = TinyTimeMixerForPrediction.from_pretrained(model_id, revision=revision).to(self.device).eval()
        self.chunk = int(self.model.config.prediction_length)

    def torch_module(self):
        return self.model

    def to_device(self, device):
        self.device = device
        self.model.to(device)
        return self

    def predict(self, context, freq, horizon):
        import torch
        outs = []
        with torch.no_grad():
            for k in range(0, len(context), self.batch):
                x = torch.from_numpy(context[k:k + self.batch]).to(self.device)
                preds, n = [], 0
                while n < horizon:
                    y = self.model(past_values=x[:, -self.L:]).prediction_outputs
                    preds.append(y)
                    x = torch.cat([x, y], dim=1)
                    n += y.shape[1]
                outs.append(torch.cat(preds, dim=1)[:, :horizon].float().cpu().numpy())
        return np.concatenate(outs)

    def info(self):
        return {"checkpoint": f"{self.model_id}@{self.revision}", "rolling_chunk": self.chunk}


class TimesFM1Forecaster(_SeriesForecaster):
    """TimesFM-1.0-200m (thesis era), timesfm 1.x API (Python <= 3.11)."""

    def __init__(self, model_id="google/timesfm-1.0-200m-pytorch", name="TimesFM-1.0", context_length=512,
                 horizon_len=720, device=None, batch=128):
        import timesfm
        self.name = name
        self.model_id = model_id
        self.batch = batch
        backend = "gpu" if _device(device).startswith("cuda") else "cpu"
        self.tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(backend=backend, per_core_batch_size=32, horizon_len=horizon_len,
                                           context_len=context_length),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=model_id))

    def torch_module(self):
        return getattr(self.tfm, "_model", None)

    def _predict_series(self, x, freq, H):
        b = bucket_name(freq)
        f = 0 if b in ("second", "minute", "10min", "15min", "30min", "hour", "day") else (1 if b in ("week", "month") else 2)
        point, _ = self.tfm.forecast(list(x), freq=[f] * len(x))
        return np.asarray(point)[:, :H]

    def info(self):
        return {"checkpoint": self.model_id}


_GLUON_FREQ = {"H": "H", "15T": "15min", "10T": "10min", "D": "D", "W": "W", "30T": "30min", "5T": "5min"}


class LagLlamaForecaster(_SeriesForecaster):
    """Lag-Llama (thesis era) through the authors' GluonTS estimator. Zero-shot settings follow the
    authors' defaults (context length 32 plus lag features); the point forecast is the sample mean."""

    def __init__(self, ckpt_repo="time-series-foundation-models/Lag-Llama", name="Lag-Llama", context_length=32,
                 num_samples=20, device=None, batch=64):
        import torch
        from huggingface_hub import hf_hub_download
        self.name = name
        self.ckpt_repo = ckpt_repo
        self.ckpt = hf_hub_download(ckpt_repo, "lag-llama.ckpt")
        self.context_length = context_length
        self.num_samples = num_samples
        self.device = _device(device)
        self.batch = batch
        self.hp = torch.load(self.ckpt, map_location="cpu", weights_only=False)["hyper_parameters"]["model_kwargs"]
        self.predictors = {}

    def _predictor(self, H):
        if H not in self.predictors:
            import torch
            from lag_llama.gluon.estimator import LagLlamaEstimator
            est = LagLlamaEstimator(
                ckpt_path=self.ckpt, prediction_length=H, context_length=self.context_length,
                input_size=self.hp["input_size"], n_layer=self.hp["n_layer"], n_embd_per_head=self.hp["n_embd_per_head"],
                n_head=self.hp["n_head"], scaling=self.hp["scaling"], time_feat=self.hp["time_feat"],
                batch_size=self.batch, num_parallel_samples=self.num_samples, device=torch.device(self.device))
            module = est.create_lightning_module()
            self.predictors[H] = est.create_predictor(est.create_transformation(), module)
        return self.predictors[H]

    def _predict_series(self, x, freq, H):
        import pandas as pd
        from gluonts.dataset.common import ListDataset
        f = _GLUON_FREQ.get(freq, freq)
        ds = ListDataset([{"start": pd.Period("2000-01-01 00:00", freq=f), "target": r.astype(np.float32)}
                          for r in x], freq=f)
        fc = list(self._predictor(H).predict(ds))
        return np.stack([np.asarray(f_.samples).mean(axis=0)[:H] for f_ in fc])

    def info(self):
        return {"checkpoint": self.ckpt_repo, "context_length": self.context_length, "num_samples": self.num_samples}


TSFM_FACTORIES = {
    # T5 decodes autoregressively with sampling: 20 samples x 256 series ran out of memory on a 16 GB T4 and
    # took ~2 min per 112 series; 10 samples x 128 series keeps Traffic (13.8k series) to ~1.5 h
    "Chronos-T": lambda **kw: ChronosForecaster("amazon/chronos-t5-tiny", "Chronos-T",
                                                **{"batch": 128, "num_samples": 10, **kw}),
    "Chronos-Bolt-T": lambda **kw: ChronosForecaster("amazon/chronos-bolt-tiny", "Chronos-Bolt-T", **kw),
    "TTM-R2": lambda **kw: TTMForecaster(name="TTM-R2", **kw),
    "TTM-R1": lambda **kw: TTMR1Forecaster(**kw),
    "TimesFM-1.0": lambda **kw: TimesFM1Forecaster(**kw),
    "Lag-Llama": lambda **kw: LagLlamaForecaster(**kw),
    "Moirai-S": lambda **kw: MoiraiForecaster(**kw),
    "TimesFM-2.5": lambda **kw: TimesFMForecaster(**kw),
}
