#!/usr/bin/env python3
"""Download the LOTSA pre-training subsets listed in configs/pretrain_corpus.yaml."""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
import yaml  # noqa: E402

from ezero.data.lotsa import download_subset  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--corpus", default=os.path.join(ROOT, "configs", "pretrain_corpus.yaml"))
p.add_argument("--root", default=os.environ.get("EZERO_LOTSA_DIR", os.path.join(ROOT, "data", "lotsa")))
a = p.parse_args()
for spec in yaml.safe_load(open(a.corpus))["subsets"]:
    print(download_subset(spec["name"], a.root, spec.get("max_shards")))
