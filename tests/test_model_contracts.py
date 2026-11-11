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


def test_mixer_gradients_finite_for_identical_channels():
    """Regression: sqrt(var) at zero group variance produced NaN gradients on GPU."""
    m = EZeroModel(small()).train()
    with torch.no_grad():
        for p in m.mixer.head.parameters():
            p.normal_(0, 0.5)
    x = torch.randn(1, 64).repeat(4, 1)            # four identical channels in one group
    y = m(x, 6, group=torch.zeros(4, dtype=torch.long))
    y.pow(2).mean().backward()
    for n, p in m.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), n


def test_truncated_horizon_is_exact():
    """Computing only the first h steps gives the same forecast as computing all and slicing."""
    m = EZeroModel(small()).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))      # make the zero-initialised heads non-trivial
        x = torch.randn(6, 64).cumsum(-1)
        g = torch.tensor([0, 0, 1, 1, 1, 2])
        full = m.forward_normalized(*m.normalize(x, torch.ones_like(x, dtype=torch.bool))[:1],
                                    torch.ones_like(x, dtype=torch.bool), torch.full((6,), 6), g)
        part = m.forward_normalized(*m.normalize(x, torch.ones_like(x, dtype=torch.bool))[:1],
                                    torch.ones_like(x, dtype=torch.bool), torch.full((6,), 6), g, out_horizon=10)
    assert torch.allclose(full[:, :10], part, atol=1e-5)


def test_linear_skip_variant():
    """Linear shortcut + N-BEATS: forward works, old configs load unchanged, parameters are added."""
    base = EZeroModel(small())
    skip = EZeroModel(small(use_linear_skip=True)).eval()
    assert skip.linear_skip is not None and base.linear_skip is None
    assert skip.num_parameters() == base.num_parameters() + (64 + 4) * 24 + 24
    x = torch.randn(5, 64).cumsum(-1)
    y = skip(x, 6, group=torch.tensor([0, 0, 1, 1, 1]))
    assert y.shape == (5, 24) and torch.isfinite(y).all()
    assert torch.allclose(skip(x, 6, horizon=10), skip(x, 6)[:, :10], atol=1e-5)
    old_cfg = {k: v for k, v in small().to_dict().items() if k != "use_linear_skip"}
    assert EZeroConfig.from_dict(old_cfg).use_linear_skip is False


# ---------------------------------------------------------------------------
# Thesis Sec. 3.3.3-3.3.5: components described in the text but absent from the
# 2024 code (docs/THESIS_REPRODUCTION_NOTES.md, section 11).

def test_faithful_variant_adds_the_three_blocks():
    base, faith = EZeroModel(small()), EZeroModel(small(**ABLATION_VARIANTS["faithful"]))
    assert base.simmatch is None and base.feat_enc is None
    assert faith.simmatch is not None and faith.feat_enc is not None and faith.mixer.attn is not None
    assert faith.tcn.network[0].film is not None and base.tcn.network[0].film is None
    assert faith.num_parameters() > base.num_parameters()
    assert EZeroModel(variant_config("faithful")).num_parameters() < 1.5e6   # thesis Table 3: ~1M


def test_feature_attention_is_permutation_invariant():
    """Zero-shot use needs the channel mixing to be independent of channel order and count."""
    m = EZeroModel(small(**ABLATION_VARIANTS["faithful"])).eval()
    with torch.no_grad():
        # the scorer starts at the unweighted mean and the mixer head at zero
        for p in m.mixer.parameters():
            p.add_(0.3 * torch.randn_like(p))
        x = torch.randn(4, 64).cumsum(-1)
        g = torch.tensor([0, 0, 0, 1])
        perm = torch.tensor([2, 0, 1, 3])
        a = m(x, freq_bucket=6, group=g)
        b = m(x[perm], freq_bucket=6, group=g[perm])
        singletons = m(x, freq_bucket=6, group=torch.tensor([0, 1, 2, 3]))
    assert torch.allclose(a[perm], b, atol=1e-5)
    assert not torch.allclose(a, singletons, atol=1e-4)   # the grouping really matters


def test_feature_attention_weights_sum_to_one_per_group():
    from ezero.models.components import grouped_softmax
    score = torch.randn(7)
    group = torch.tensor([0, 0, 2, 2, 2, 3, 3])
    alpha = grouped_softmax(score, group, 4)
    total = torch.zeros(4).index_add_(0, group, alpha)
    assert torch.allclose(total[[0, 2, 3]], torch.ones(3), atol=1e-6)
    assert (alpha > 0).all()


def test_similarity_matching_uses_only_the_context():
    """Adaptive inference must not see the horizon: the embedding is a function of the context."""
    m = EZeroModel(small(**ABLATION_VARIANTS["faithful"])).eval()
    x = torch.randn(3, 64).cumsum(-1)
    obs = torch.ones_like(x, dtype=torch.bool)
    with torch.no_grad():
        e1 = m.simmatch(*m.normalize(x, obs)[:1], obs)
        e2 = m.simmatch(*m.normalize(x, obs)[:1], obs)
    assert e1.shape == (3, m.config.sim_dim) and torch.allclose(e1, e2)
    assert torch.allclose(m(x, 6, horizon=10), m(x, 6)[:, :10], atol=1e-5)


def test_conditional_convolution_starts_as_identity():
    """FiLM is zero-initialised, so adding it does not change an already trained block."""
    torch.manual_seed(0)
    plain = TemporalConvNet(3, [8, 8], 3, [1, 2], dropout=0.0).eval()
    torch.manual_seed(0)
    cond = TemporalConvNet(3, [8, 8], 3, [1, 2], dropout=0.0, cond_dim=5).eval()
    cond.load_state_dict(plain.state_dict(), strict=False)
    x, c = torch.randn(4, 3, 20), torch.randn(4, 5)
    with torch.no_grad():
        assert torch.allclose(plain(x), cond(x, c), atol=1e-6)
        for p in cond.network[0].film.parameters():
            p.add_(0.5 * torch.randn_like(p))
        assert not torch.allclose(plain(x), cond(x, c), atol=1e-4)
        # the filters really depend on the conditioning vector
        assert not torch.allclose(cond(x, c), cond(x, torch.randn(4, 5)), atol=1e-4)


def test_old_configs_default_the_new_flags_off():
    old = {k: v for k, v in small().to_dict().items()
           if k not in ("use_feat_attn", "use_condconv", "use_simmatch", "feat_dim", "sim_dim", "n_prototypes")}
    cfg = EZeroConfig.from_dict(old)
    assert not (cfg.use_feat_attn or cfg.use_condconv or cfg.use_simmatch)


def test_faithful_variant_gradients_finite_for_identical_channels():
    m = EZeroModel(small(**ABLATION_VARIANTS["faithful"]))
    x = torch.zeros(4, 64)
    x[2:] = torch.randn(2, 64).cumsum(-1)
    m(x, freq_bucket=6, group=torch.zeros(4, dtype=torch.long)).pow(2).mean().backward()
    for n, p in m.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), n
