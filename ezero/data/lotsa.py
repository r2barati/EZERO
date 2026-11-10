"""LOTSA (Salesforce/lotsa_data) pre-training corpus.

Each LOTSA sub-dataset is a HuggingFace ``save_to_disk`` folder with records
``{item_id, start, freq, target, [past_feat_dynamic_real]}`` where ``target`` is
1-D (univariate) or 2-D ``(channels, time)``.

Records are converted to *panels* - float32 arrays of shape (C, T):

* a 2-D record is one panel (its variates are time-aligned);
* univariate records of one sub-dataset that share ``start`` and length (sensor
  networks such as LOS_LOOP or PEMS) are stacked into one panel, so the
  Exogenous Mixer sees realistic multi-channel groups during pre-training.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import numpy as np

from ezero.frequency import bucket_id

LOTSA_REPO = "Salesforce/lotsa_data"

# LOTSA subsets that share a source with the zero-shot evaluation datasets. Pre-training on
# them would leak test information (e.g. Monash ``traffic_hourly`` is the PeMS-SF data behind
# the LSF Traffic benchmark), so ``load_corpus`` refuses them unless explicitly allowed.
EVAL_OVERLAP_BLOCKLIST = {
    "traffic_hourly": "Traffic (PeMS-SF occupancy)",
    "traffic_weekly": "Traffic (PeMS-SF occupancy)",
    "electricity_hourly": "Electricity (UCI ElectricityLoadDiagrams)",
    "electricity_weekly": "Electricity (UCI ElectricityLoadDiagrams)",
    "ett_h1": "ETTh1", "ett_h2": "ETTh2", "ett_m1": "ETTm1", "ett_m2": "ETTm2",
}


@dataclass
class Panel:
    dataset: str
    freq: str
    values: np.ndarray  # (C, T) float32, NaN = missing

    @property
    def length(self) -> int:
        return self.values.shape[1]

    @property
    def channels(self) -> int:
        return self.values.shape[0]

    @property
    def bucket(self) -> int:
        return bucket_id(self.freq)


def download_subset(name: str, root: str, max_shards: int | None = None) -> str:
    """Download one LOTSA sub-dataset folder (optionally only the first ``max_shards`` arrow shards)."""
    from huggingface_hub import HfApi, hf_hub_download

    target = os.path.join(root, name)
    if os.path.exists(os.path.join(target, "state.json")) and glob.glob(os.path.join(target, "*.arrow")):
        return target
    api = HfApi()
    files = [f for f in api.list_repo_files(LOTSA_REPO, repo_type="dataset") if f.startswith(name + "/")]
    if not files:
        raise FileNotFoundError(f"LOTSA sub-dataset {name!r} not found in {LOTSA_REPO}")
    arrows = sorted(f for f in files if f.endswith(".arrow"))
    keep_arrows = arrows if max_shards is None else arrows[:max_shards]
    for f in files:
        if f.endswith(".arrow") and f not in keep_arrows:
            continue
        hf_hub_download(LOTSA_REPO, f, repo_type="dataset", local_dir=root)
    if max_shards is not None and len(keep_arrows) < len(arrows):
        # rewrite state.json so load_from_disk only expects the shards we kept
        state_path = os.path.join(target, "state.json")
        with open(state_path) as fh:
            state = json.load(fh)
        kept = {os.path.basename(a) for a in keep_arrows}
        state["_data_files"] = [d for d in state.get("_data_files", []) if d.get("filename") in kept]
        with open(state_path, "w") as fh:
            json.dump(state, fh)
    return target


def _iter_records(path: str):
    import datasets

    try:
        ds = datasets.load_from_disk(path)
    except Exception:  # fall back to reading the arrow shards directly
        shards = sorted(glob.glob(os.path.join(path, "*.arrow")))
        ds = datasets.concatenate_datasets([datasets.Dataset.from_file(s) for s in shards])
    ds = ds.select_columns([c for c in ("start", "freq", "target") if c in ds.column_names])
    for rec in ds:
        yield rec


def load_panels(name: str, path: str, max_series: int | None = None, min_length: int = 64,
                max_length: int | None = None, max_group: int = 64) -> list[Panel]:
    """Read one sub-dataset into panels."""
    panels: list[Panel] = []
    uni: dict[tuple, list[np.ndarray]] = {}
    freq_seen = None
    n = 0
    for rec in _iter_records(path):
        tgt = np.asarray(rec["target"], dtype=np.float32)
        freq = rec.get("freq") or freq_seen or ""
        freq_seen = freq
        if max_length is not None and tgt.shape[-1] > max_length:
            tgt = tgt[..., -max_length:]
        if tgt.shape[-1] < min_length:
            continue
        if tgt.ndim == 2:
            panels.append(Panel(name, freq, np.ascontiguousarray(tgt)))
        else:
            key = (str(rec.get("start")), tgt.shape[0])
            uni.setdefault(key, []).append(tgt)
        n += 1
        if max_series is not None and n >= max_series:
            break
    for (_, _), arrs in uni.items():
        for i in range(0, len(arrs), max_group):
            panels.append(Panel(name, freq_seen or "", np.stack(arrs[i:i + max_group])))
    return panels


def check_no_eval_overlap(names) -> None:
    bad = {n: EVAL_OVERLAP_BLOCKLIST[n.lower()] for n in names if n.lower() in EVAL_OVERLAP_BLOCKLIST}
    if bad:
        raise ValueError(f"pre-training corpus overlaps the evaluation data: {bad}")


def load_corpus(subsets: list[dict], root: str, download: bool = True, verbose: bool = True,
                allow_eval_overlap: bool = False) -> list[Panel]:
    """Load all sub-datasets listed in the corpus config (see ``configs/pretrain_corpus.yaml``)."""
    if not allow_eval_overlap:
        check_no_eval_overlap([s["name"] for s in subsets])
    panels: list[Panel] = []
    for spec in subsets:
        name = spec["name"]
        path = os.path.join(root, name)
        if download:
            path = download_subset(name, root, spec.get("max_shards"))
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{path} missing; run scripts/data/download_lotsa.py first")
        ps = load_panels(name, path, spec.get("max_series"), spec.get("min_length", 64),
                         spec.get("max_length"), spec.get("max_group", 64))
        if spec.get("freq"):
            for p in ps:
                p.freq = spec["freq"]
        if verbose:
            obs = sum(int(np.isfinite(p.values).sum()) for p in ps)
            print(f"[lotsa] {name:32s} panels={len(ps):6d} channels={sum(p.channels for p in ps):7d} "
                  f"obs={obs / 1e6:8.2f}M freq={ps[0].freq if ps else '?'}", flush=True)
        panels.extend(ps)
    return panels
