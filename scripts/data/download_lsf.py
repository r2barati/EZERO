#!/usr/bin/env python3
"""Download / locate the seven LSF evaluation CSVs and verify shape + SHA-256."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from ezero.data.ltsf import LSF_DATASETS, _sha256, fetch, load_lsf  # noqa: E402

ok = True
for name, spec in LSF_DATASETS.items():
    path = fetch(name)
    sha = _sha256(path)
    d = load_lsf(name, path=path)
    good = sha == spec["sha256"]
    ok &= good
    print(f"{name:12s} {d.values.shape} sha256 {'OK' if good else 'MISMATCH ' + sha} {path}")
sys.exit(0 if ok else 1)
