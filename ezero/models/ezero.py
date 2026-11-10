"""E-ZERO hybrid model: N-BEATS (with CRRB) -> TCN (with MRC + DTP) -> Exogenous Mixer.

Forward pass for a batch of univariate series (channel-independent backbone):

1. Instance normalisation using only the observed context (no train/test leakage).
2. RAE: frequency-bucket embedding, used as a conditioning vector by N-BEATS and as
   constant input channels of the TCN.
3. N-BEATS stack (``nbeats_blocks`` blocks, CRRB paths) -> coarse forecast (Eq. 9, 17).
4. TCN refinement over the sequence [context ; coarse forecast] (Eq. 10, 13): MRC input
   layer, causal dilated TemporalBlocks, DTP, 1x1 head producing a forecast correction.
5. Exogenous Mixer across the channels of each group (Eq. 22-26).

All forecasts are produced for ``horizon`` (default 720) steps; shorter horizons are a
prefix of the output, so one checkpoint serves the 96/192/336/720 protocol.

Ablation flags reproduce the thesis ablation study (Table 5): ``use_mrc``, ``use_dtp``,
``use_rae``, ``use_crrb``, ``use_exog``, ``use_tcn``, ``use_nbeats``. Without N-BEATS the
coarse forecast comes from a single linear projection of the context; without TCN the
N-BEATS forecast is used directly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace

import torch
import torch.nn as nn

from ezero.models.components import (DynamicTemporalPooling, ExogenousMixer, MultiResolutionConv,
                                     ResolutionEmbedding)
from ezero.models.nbeats import CrossResolutionBlock, NBeatsBlock
from ezero.models.tcn import CausalConv1d, TemporalConvNet


@dataclass
class EZeroConfig:
    # sequence lengths
    context_length: int = 512
    horizon: int = 720
    # N-BEATS (thesis final selection: 3 blocks x 64 neurons)
    nbeats_blocks: int = 3
    nbeats_hidden: int = 64
    nbeats_layers: int = 4
    crrb_resolutions: tuple = (1, 2, 4)
    # TCN (thesis final selection: 4 layers, 32 filters, dilations 1,2,4,8, kernel 3)
    tcn_channels: int = 32
    tcn_dilations: tuple = (1, 2, 4, 8)
    tcn_kernel_size: int = 3
    mrc_dilations: tuple = (1, 2, 4, 8)
    # misc
    dropout: float = 0.2
    rae_dim: int = 8
    mixer_channels: int = 16
    mixer_layers: int = 2
    norm_eps: float = 1e-5
    target_clip: float = 10.0
    # ablation switches
    use_mrc: bool = True
    use_dtp: bool = True
    use_rae: bool = True
    use_crrb: bool = True
    use_exog: bool = True
    use_tcn: bool = True
    use_nbeats: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EZeroConfig":
        names = {f.name for f in fields(cls)}
        kw = {}
        for k, v in d.items():
            if k not in names:
                continue
            kw[k] = tuple(v) if isinstance(v, list) else v
        return cls(**kw)


# Thesis Table 5 columns -> config overrides
ABLATION_VARIANTS = {
    "base": {},
    "noMRC": {"use_mrc": False},
    "noDTP": {"use_dtp": False},
    "noRAE": {"use_rae": False},
    "noCRRB": {"use_crrb": False},
    "noExog": {"use_exog": False},
    "noTCN": {"use_tcn": False},
    "noNBEATS": {"use_nbeats": False},
}


def variant_config(variant: str = "base", **overrides) -> EZeroConfig:
    if variant not in ABLATION_VARIANTS:
        raise KeyError(f"unknown variant {variant!r}; choose from {list(ABLATION_VARIANTS)}")
    return replace(EZeroConfig(**overrides), **ABLATION_VARIANTS[variant])


class EZeroModel(nn.Module):
    def __init__(self, config: EZeroConfig | None = None):
        super().__init__()
        cfg = config or EZeroConfig()
        if not (cfg.use_nbeats or cfg.use_tcn):
            raise ValueError("at least one of use_nbeats / use_tcn must be enabled")
        self.config = cfg
        L, H = cfg.context_length, cfg.horizon
        cond_dim = cfg.rae_dim if cfg.use_rae else 0
        self.rae = ResolutionEmbedding(cfg.rae_dim) if cfg.use_rae else None

        if cfg.use_nbeats:
            if cfg.use_crrb:
                blocks = [CrossResolutionBlock(L, H, cfg.nbeats_hidden, cfg.nbeats_layers, cfg.dropout,
                                               cond_dim, cfg.crrb_resolutions) for _ in range(cfg.nbeats_blocks)]
            else:
                blocks = [NBeatsBlock(L, H, cfg.nbeats_hidden, cfg.nbeats_layers, cfg.dropout, cond_dim)
                          for _ in range(cfg.nbeats_blocks)]
            self.blocks = nn.ModuleList(blocks)
            self.linear_fallback = None
        else:
            self.blocks = None
            self.linear_fallback = nn.Linear(L + cond_dim, H)

        if cfg.use_tcn:
            in_ch = 3 + cond_dim  # value, observed flag, forecast-slot flag (+ RAE channels)
            if cfg.use_mrc:
                self.tcn_in = MultiResolutionConv(in_ch, cfg.tcn_channels, cfg.tcn_kernel_size, cfg.mrc_dilations)
            else:
                self.tcn_in = nn.Sequential(CausalConv1d(in_ch, cfg.tcn_channels, cfg.tcn_kernel_size), nn.ReLU())
            self.tcn = TemporalConvNet(cfg.tcn_channels, [cfg.tcn_channels] * len(cfg.tcn_dilations),
                                       cfg.tcn_kernel_size, list(cfg.tcn_dilations), cfg.dropout)
            self.dtp = DynamicTemporalPooling(cfg.tcn_channels) if cfg.use_dtp else None
            self.tcn_head = nn.Conv1d(cfg.tcn_channels, 1, 1)
            nn.init.zeros_(self.tcn_head.weight)
            nn.init.zeros_(self.tcn_head.bias)
        else:
            self.tcn_in = self.tcn = self.dtp = self.tcn_head = None

        self.mixer = ExogenousMixer(cfg.mixer_channels, cfg.mixer_layers) if cfg.use_exog else None

    # ------------------------------------------------------------------
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def normalize(self, context: torch.Tensor, observed: torch.Tensor):
        obs = observed.to(context.dtype)
        n = obs.sum(-1, keepdim=True).clamp(min=1.0)
        mu = (context * obs).sum(-1, keepdim=True) / n
        var = (((context - mu) * obs) ** 2).sum(-1, keepdim=True) / n
        sigma = var.sqrt() + self.config.norm_eps
        x = ((context - mu) / sigma) * obs
        return x, mu, sigma

    def forward_normalized(self, x: torch.Tensor, observed: torch.Tensor, freq_bucket: torch.Tensor,
                           group: torch.Tensor | None = None, exog_mask: torch.Tensor | None = None,
                           exog_future: torch.Tensor | None = None, out_horizon: int | None = None) -> torch.Tensor:
        """Forecast in the instance-normalised space. x: (N, L) normalised, zero where unobserved.

        ``out_horizon`` < horizon computes only the first ``out_horizon`` steps. This is exact: every TCN,
        MRC, DTP and mixer operation after N-BEATS is causal along time, so step t never depends on
        later steps (tests/test_model_contracts.py checks equality)."""
        cfg = self.config
        N, L = x.shape
        H = cfg.horizon
        cond = self.rae(freq_bucket) if self.rae is not None else None

        if self.blocks is not None:
            residual = x
            forecast = x.new_zeros(N, H)
            for block in self.blocks:
                b, f = block(residual, cond)
                residual = residual - b
                forecast = forecast + f
        else:
            inp = x if cond is None else torch.cat([x, cond], dim=-1)
            forecast = self.linear_fallback(inp)

        if out_horizon is not None and out_horizon < H:
            forecast = forecast[:, :out_horizon]
            if exog_future is not None:
                exog_future = exog_future[:, :out_horizon]
            H = out_horizon

        if self.tcn is not None:
            value = torch.cat([x, forecast], dim=-1)                                   # (N, L+H)
            obs_flag = torch.cat([observed.to(x.dtype), x.new_zeros(N, H)], dim=-1)
            fc_flag = torch.cat([x.new_zeros(N, L), x.new_ones(N, H)], dim=-1)
            chans = [value, obs_flag, fc_flag]
            seq = torch.stack(chans, dim=1)                                            # (N, 3, L+H)
            if cond is not None:
                seq = torch.cat([seq, cond.unsqueeze(-1).expand(-1, -1, L + H)], dim=1)
            h = self.tcn(self.tcn_in(seq))
            if self.dtp is not None:
                h = self.dtp(h, freq_bucket)
            forecast = forecast + self.tcn_head(h[..., L:]).squeeze(1)

        if self.mixer is not None and group is not None:
            forecast = self.mixer(forecast, group, exog_mask, exog_future)
        return forecast

    def forward(self, context: torch.Tensor, freq_bucket: torch.Tensor | int | None = None,
                observed: torch.Tensor | None = None, group: torch.Tensor | None = None,
                horizon: int | None = None, exog_mask: torch.Tensor | None = None,
                exog_future: torch.Tensor | None = None) -> torch.Tensor:
        """Forecast ``horizon`` steps for each row of ``context`` (N, L) in the original scale.

        ``group`` assigns rows to multivariate groups (rows of the same series/window); rows
        sharing a group id are mixed by the Exogenous Mixer. ``observed`` marks valid context
        values (NaNs are treated as unobserved automatically). Contexts shorter than
        ``context_length`` are left-padded, longer ones are truncated to the last values.
        """
        cfg = self.config
        if context.dim() != 2:
            raise ValueError("context must be (N, L)")
        N = context.shape[0]
        if observed is None:
            observed = torch.isfinite(context)
        context = torch.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
        L = cfg.context_length
        if context.shape[1] > L:
            context, observed = context[:, -L:], observed[:, -L:]
        elif context.shape[1] < L:
            pad = L - context.shape[1]
            context = torch.cat([context.new_zeros(N, pad), context], dim=1)
            observed = torch.cat([observed.new_zeros(N, pad), observed], dim=1)
        if freq_bucket is None:
            freq_bucket = 0
        if not torch.is_tensor(freq_bucket):
            freq_bucket = torch.full((N,), int(freq_bucket), dtype=torch.long, device=context.device)
        x, mu, sigma = self.normalize(context, observed)
        ef = None
        if exog_future is not None:
            ef = ((exog_future - mu) / sigma).clamp(-cfg.target_clip, cfg.target_clip)
        y = self.forward_normalized(x, observed, freq_bucket, group, exog_mask, ef, out_horizon=horizon)
        y = y * sigma + mu
        if horizon is not None:
            y = y[:, :horizon]
        return y
