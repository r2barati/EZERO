"""Shape, causality and ablation contracts of the E-ZERO model."""
import pytest
import torch

from ezero import ABLATION_VARIANTS, EZeroConfig, EZeroModel, load_checkpoint, save_checkpoint, variant_config
from ezero.models.tcn import TemporalConvNet


def small(**kw):
    return EZeroConfig(context_length=64, horizon=24, nbeats_hidden=16, tcn_channels=8, rae_dim=4,
                       mixer_channels=4, **kw)


@pytest.mark.parametrize("variant", list(ABLATION_VARIANTS))
def test_variants_forward(variant):
    cfg = EZeroConfig.from_dict({**small().to_dict(), **ABLATION_VARIANTS[variant]})
    m = EZeroModel(cfg).eval()
    x = torch.randn(6, 64).cumsum(-1)
    y = m(x, freq_bucket=6, group=torch.tensor([0, 0, 0, 1, 1, 2]))
    assert y.shape == (6, 24)
    assert torch.isfinite(y).all()


def test_full_size_parameter_budget():
    """Thesis Table 3 reports ~1M parameters for E-ZERO."""
    n = EZeroModel(variant_config("base")).num_parameters()
    assert 0.3e6 < n < 1.5e6


def test_horizon_prefix_and_short_context():
    m = EZeroModel(small()).eval()
    x = torch.randn(3, 40)            # shorter than context_length -> left padded
    full = m(x, 6)
    pre = m(x, 6, horizon=10)
    assert torch.allclose(full[:, :10], pre)


def test_scale_equivariance():
    """Instance normalisation: forecasting a*x+b gives a*f(x)+b."""
    m = EZeroModel(small()).eval()
    x = torch.randn(4, 64).cumsum(-1)
    y1 = m(x, 6)
    y2 = m(3.0 * x + 7.0, 6)
    assert torch.allclose(y2, 3.0 * y1 + 7.0, atol=1e-3, rtol=1e-3)


def test_nan_context_is_handled():
    m = EZeroModel(small()).eval()
    x = torch.randn(2, 64)
    x[0, :10] = float("nan")
    assert torch.isfinite(m(x, 6)).all()


def test_tcn_is_causal():
    net = TemporalConvNet(1, [4, 4], kernel_size=3).eval()
    x = torch.randn(1, 1, 50)
    y = net(x)
    x2 = x.clone()
    x2[..., 30:] += 10.0
    y2 = net(x2)
    assert torch.allclose(y[..., :30], y2[..., :30])


def test_mixer_only_mixes_within_groups():
    m = EZeroModel(small()).eval()
    x = torch.randn(4, 64)
    g = torch.tensor([0, 0, 1, 1])
    with torch.no_grad():
        for p in m.mixer.head.parameters():
            p.normal_(0, 0.5)
    y = m(x, 6, group=g)
    x2 = x.clone()
    x2[2:] = torch.randn(2, 64) * 5
    y2 = m(x2, 6, group=g)
    assert torch.allclose(y[:2], y2[:2], atol=1e-5)


def test_exogenous_future_values_are_passed_through():
    m = EZeroModel(small()).eval()
    x = torch.randn(3, 64)
    fut = torch.randn(3, 24)
    mask = torch.tensor([False, True, False])
    y = m(x, 6, group=torch.zeros(3, dtype=torch.long), exog_mask=mask, exog_future=fut)
    assert torch.allclose(y[1], fut[1], atol=1e-4)


def test_checkpoint_roundtrip(tmp_path):
    m = EZeroModel(small()).eval()
    p = tmp_path / "m.pt"
    save_checkpoint(m, str(p), extra={"note": "x"})
    m2 = load_checkpoint(str(p))
    x = torch.randn(2, 64)
    assert torch.allclose(m(x, 6), m2(x, 6))
    assert m2.checkpoint_extra["note"] == "x"


def test_both_backbones_disabled_is_rejected():
    with pytest.raises(ValueError):
        EZeroModel(small(use_nbeats=False, use_tcn=False))
