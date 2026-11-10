"""LSF splits / scaling / window protocol (uses data/raw when present)."""
import numpy as np
import pytest

from ezero.data.ltsf import LSF_DATASETS, compute_borders, find_file, load_lsf, window_starts
from ezero.data.pretrain import WindowSampler, synthetic_panels


def have(name):
    return find_file(LSF_DATASETS[name]["file"]) is not None


def test_borders_match_time_series_library():
    b = compute_borders(17420, "ett_hour", 512)
    assert b == {"train": (0, 8640), "val": (8128, 11520), "test": (11008, 14400)}
    b = compute_borders(26304, "ratio", 512)
    assert b["train"] == (0, 18412) and b["test"] == (26304 - 5260 - 512, 26304)


@pytest.mark.skipif(not have("ETTh1"), reason="ETTh1.csv not available")
def test_etth1_standard_window_count_and_train_scaling():
    d = load_lsf("ETTh1")
    assert d.n_windows(96) == 2785 and d.n_windows(720) == 2161
    tr = d.split_values("train")
    assert np.allclose(tr.mean(0), 0, atol=1e-4) and np.allclose(tr.std(0), 1, atol=1e-3)
    full = load_lsf("ETTh1", scaler="full")
    assert not np.allclose(full.mean, d.mean)


@pytest.mark.skipif(not have("ETTh1"), reason="ETTh1.csv not available")
def test_subsample_protocol_is_shared_across_horizons():
    d = load_lsf("ETTh1")
    s = window_starts(d, "sub16")
    assert len(s) == 16 and s.max() <= d.n_windows(720) - 1
    assert len(window_starts(d, "full")) == d.n_windows(96)


def test_sampler_targets_never_cross_split():
    panels = synthetic_panels(4, 1000)
    s = WindowSampler(panels, 64, 24, val_fraction=0.2, seed=0)
    b = s.sample(32)
    assert b["context"].shape == (len(b["group"]), 64)
    assert b["target"].shape[1] == 24
    # train targets are cut at the split -> the tail values never appear in train batches
    for p, cut in zip(panels, s.split):
        assert cut <= int(p.length * 0.8) + 1
    vb = s.sample(16, "val")
    assert np.isfinite(vb["target"]).any()
