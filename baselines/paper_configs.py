"""Official per-dataset settings of the full-shot baselines, for the reproduction check only.

Values were read from the authors' run scripts (or, where the authors ship none, from the paper /
Time-Series-Library scripts) for ETTh1 and ETTm1. Used by ``run_benchmarks.py --paper-config``;
results are stored under protocol ``fullpaper`` (same test windows as ``full``) and never mix with
the canonical lookback-512 runs. Sources:

* iTransformer  github.com/thuml/iTransformer scripts/multivariate_forecasting/ETT/iTransformer_ETT{h,m}1.sh
* TimesNet      github.com/thuml/Time-Series-Library scripts/long_term_forecast/ETT_script/TimesNet_ETT{h,m}1.sh
* TimeMixer     github.com/kwuking/TimeMixer scripts/long_term_forecast/ETT_script/TimeMixer_ETT{h,m}1_unify.sh
* FEDformer     github.com/MAZiqing/FEDformer scripts/run_M.sh (+ run.py defaults: modes 64, moving_avg 24,
                dropout 0.05, d_model 512, d_ff 2048)
* Autoformer    github.com/thuml/Time-Series-Library ETT_script/Autoformer_ETT{h,m}1.sh (the authors' own
                ETT scripts use Informer horizons); d_model 512, d_ff 2048, factor 3, dropout 0.1
* PatchTST      github.com/yuqinie98/PatchTST PatchTST_supervised/scripts/PatchTST/ett{h,m}1.sh (= PatchTST/42)
* DLinear       github.com/cure-lab/LTSF-Linear scripts/EXP-LongForecasting/Linear/ett{h,m}1.sh
* TSMixer       paper (arXiv 2303.06053v5) Table 8 + tsmixer_basic/run.py defaults (batch 32, BatchNorm,
                RevIN, 100 epochs, patience 5, constant LR); the authors' script has no ETTh1/ETTm1 entry

Time-stamp ("time feature") embeddings used by iTransformer/TimesNet/FEDformer/Autoformer are not used
here (no model in this repository consumes calendar features); this is a known residual difference.
"""
from __future__ import annotations

_T1 = dict(lradj="type1", epochs=10, patience=3, batch=32)

PAPER_CONFIGS = {
    # (arch, dataset): dict(lookback, model kwargs (optionally per horizon), training settings)
    ("DLinear", "ETTh1"): dict(lookback=336, model={}, train=dict(_T1, lr=0.005)),
    ("DLinear", "ETTm1"): dict(lookback=336, model={}, train=dict(_T1, lr=1e-4, batch=8)),

    ("PatchTST", "ETTh1"): dict(lookback=336, model=dict(heads=4, d_model=16, ff=128, dropout=0.3),
                                train=dict(lr=1e-4, lradj="type3", epochs=100, patience=100, batch=128)),
    ("PatchTST", "ETTm1"): dict(lookback=336, model=dict(heads=16, d_model=128, ff=256, dropout=0.2),
                                train=dict(lr=1e-4, lradj="TST", pct_start=0.4, epochs=100, patience=20, batch=128)),

    ("iTransformer", "ETTh1"): dict(lookback=96,
                                    model={96: dict(d_model=256, ff=256), 192: dict(d_model=256, ff=256),
                                           336: dict(d_model=512, ff=512), 720: dict(d_model=512, ff=512)},
                                    train=dict(_T1, lr=1e-4)),
    ("iTransformer", "ETTm1"): dict(lookback=96, model=dict(d_model=128, ff=128), train=dict(_T1, lr=1e-4)),

    ("TimesNet", "ETTh1"): dict(lookback=96, model=dict(d_model=16, d_ff=32), train=dict(_T1, lr=1e-4)),
    ("TimesNet", "ETTm1"): dict(lookback=96,
                                model={96: dict(d_model=64, d_ff=64), 192: dict(d_model=64, d_ff=64),
                                       336: dict(d_model=16, d_ff=32), 720: dict(d_model=16, d_ff=32)},
                                train={"default": dict(_T1, lr=1e-4), 336: dict(_T1, lr=1e-4, epochs=3)}),

    ("TimeMixer", "ETTh1"): dict(lookback=96, model=dict(d_model=16, d_ff=32),
                                 train=dict(lr=0.01, lradj="TST", pct_start=0.2, epochs=10, patience=10, batch=128)),
    ("TimeMixer", "ETTm1"): dict(lookback=96, model=dict(d_model=16, d_ff=32),
                                 train=dict(lr=0.01, lradj="TST", pct_start=0.2, epochs=100, patience=15, batch=16)),

    ("FEDformer", "ETTh1"): dict(lookback=96, model=dict(d_model=512, d_ff=2048, modes=64, kernel=24, dropout=0.05,
                                                         use_norm=False), train=dict(_T1, lr=1e-4)),
    ("FEDformer", "ETTm1"): dict(lookback=96, model=dict(d_model=512, d_ff=2048, modes=64, kernel=24, dropout=0.05,
                                                         use_norm=False), train=dict(_T1, lr=1e-4)),

    ("Autoformer", "ETTh1"): dict(lookback=96, model=dict(d_model=512, d_ff=2048, factor=3, kernel=25, dropout=0.1,
                                                          use_norm=False), train=dict(_T1, lr=1e-4)),
    ("Autoformer", "ETTm1"): dict(lookback=96, model=dict(d_model=512, d_ff=2048, factor=3, kernel=25, dropout=0.1,
                                                          use_norm=False), train=dict(_T1, lr=1e-4)),

    ("TSMixer", "ETTh1"): dict(lookback=512, model={96: dict(paper=True, n_block=6, ff_dim=512, dropout=0.9),
                                                    192: dict(paper=True, n_block=4, ff_dim=256, dropout=0.9),
                                                    336: dict(paper=True, n_block=4, ff_dim=256, dropout=0.9),
                                                    720: dict(paper=True, n_block=2, ff_dim=64, dropout=0.9)},
                               train={96: dict(lr=1e-4, lradj="constant", epochs=100, patience=5, batch=32),
                                      "default": dict(lr=1e-3, lradj="constant", epochs=100, patience=5, batch=32)}),
    ("TSMixer", "ETTm1"): dict(lookback=512, model={96: dict(paper=True, n_block=6, ff_dim=16, dropout=0.9),
                                                    192: dict(paper=True, n_block=4, ff_dim=32, dropout=0.9),
                                                    336: dict(paper=True, n_block=4, ff_dim=64, dropout=0.9),
                                                    720: dict(paper=True, n_block=4, ff_dim=16, dropout=0.9)},
                               train=dict(lr=1e-4, lradj="constant", epochs=100, patience=5, batch=32)),
}


def lookup(arch: str, dataset: str, horizon: int):
    """(lookback, model kwargs, train settings) for a paper-config run, or None."""
    cfg = PAPER_CONFIGS.get((arch, dataset))
    if cfg is None:
        return None

    def pick(d):
        if any(isinstance(k, int) for k in d):
            return d.get(horizon, d.get("default", {}))
        return d.get("default", d) if "default" in d else d

    return cfg["lookback"], dict(pick(cfg["model"])), dict(pick(cfg["train"]))
