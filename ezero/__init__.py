"""E-ZERO: a more efficient (non-attention) foundation model for zero-shot forecasting.

Hybrid N-BEATS + TCN architecture with the multi-resolution enhancements described
in the thesis (MRC, DTP, RAE, CRRB and the Exogenous Mixer block).

Public API::

    from ezero import EZeroConfig, EZeroModel, load_checkpoint
"""
from ezero.models.ezero import EZeroConfig, EZeroModel, ABLATION_VARIANTS, variant_config
from ezero.checkpoint import load_checkpoint, save_checkpoint

__version__ = "2.0.0"

__all__ = [
    "EZeroConfig",
    "EZeroModel",
    "ABLATION_VARIANTS",
    "variant_config",
    "load_checkpoint",
    "save_checkpoint",
    "__version__",
]
