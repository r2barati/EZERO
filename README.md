# E-ZERO: A More Efficient Foundation Model for Zero-Shot Forecasting

Code, checkpoints and benchmark artifacts for the M.Sc. thesis *E-ZERO: A More Efficient
Foundation Model for Zero-Shot Forecasting* (Reza Barati, Toronto Metropolitan University, 2024).

E-ZERO is a non-transformer, non-attention hybrid **N-BEATS + TCN** model (about 0.74M parameters),
pre-trained on the LOTSA archive and evaluated **zero-shot** on the standard long-sequence
forecasting (LSF) benchmarks. It includes the multi-resolution enhancements described in thesis
Sec. 3.5: Multi-Resolution Convolutions (MRC), Dynamic Temporal Pooling (DTP), Resolution-Aware
Embeddings (RAE), Cross-Resolution Residual Blocks (CRRB) and the Exogenous Mixer block.

The repository follows the same layout and workflow as the `gym-invmgmt` (Paper 1) release:
a core package, a registry of evaluated methods, a resumable benchmark CLI with per-method
caches, one canonical merged CSV, a manifest, figure/table regeneration scripts, a test
suite and a publication check. Everything can be run on Kaggle GPUs with the kernels in `kaggle/`.

## Repository structure

- `ezero/`: the core package.
  - `models/`: `tcn.py` (causal dilated TCN), `nbeats.py` (N-BEATS blocks + CRRB),
    `components.py` (MRC, DTP, RAE, Exogenous Mixer), `ezero.py` (`EZeroModel`, `EZeroConfig`,
    the 8 ablation variants).
  - `data/`: `lotsa.py` (LOTSA download → multi-channel panels), `pretrain.py` (frequency-balanced
    window sampler with a held-out validation tail), `ltsf.py` (ETT / Weather / Electricity /
    Traffic loaders, standard splits, train-split scaling, window protocols).
  - `training.py` (pre-training loop), `evaluation.py` (streaming MSE/MAE), `efficiency.py`
    (inference cost + efficiency score), `checkpoint.py`, `frequency.py`.
- `baselines/`: every evaluated method behind one `predict(context, freq, horizon)` interface
  (the analogue of Paper 1's `agents/`): E-ZERO checkpoints, naive baselines, full-shot
  DLinear / PatchTST / iTransformer, and zero-shot Chronos / TTM / Moirai / TimesFM-2.5 wrappers.
- `benchmarks/`: `pretrain.py`, `run_benchmarks.py` (registry of **19 models**, caching, merge,
  manifest), `run_efficiency.py`, `verify_publication_ready.py`.
- `configs/`: thesis hyperparameters (`ezero_base.yaml`), LOTSA corpus (`pretrain_corpus.yaml`),
  evaluation protocol (`eval_lsf.yaml`).
- `kaggle/`: kernel runner, kernel builder/pusher and output fetcher.
- `results/`: benchmark outputs.
  - `cache/`: per-model CSV caches (`bench_<model>.csv`, `efficiency_<model>.csv`).
  - `benchmark_final_merged.csv`: the canonical wide table consumed by the figure scripts.
  - `benchmark_long.csv`, `efficiency.csv`, `manifest.json`, `training_logs/`, `kaggle_logs/`.
- `data/models/`: E-ZERO checkpoints (`ezero_<variant>.pt`, a few MB each).
- `paper/figures/`, `paper/tables/`: regenerated thesis figures and tables.
- `tests/`: model, protocol and runner contracts.

## Installation

```bash
pip install -e ".[dev]"
```

The foundation-model baselines are optional extras with conflicting pins; install them one at a
time (the Kaggle `tsfm` stage gives each its own virtualenv):

```bash
pip install -e ".[chronos]"   # or [ttm], [moirai], [timesfm]
```

## Data

- **Pre-training:** 10 LOTSA sub-datasets (`configs/pretrain_corpus.yaml`, carried over from the
  original EZERO data-loader config), downloaded from `Salesforce/lotsa_data` on first use.
  None of the evaluation datasets is in the corpus, so evaluation is zero-shot.
- **Evaluation:** ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity, Traffic.

```bash
python scripts/data/download_lsf.py      # fetches and checks shape + SHA-256 of all 7 CSVs
python scripts/data/download_lotsa.py    # optional: pre-download the LOTSA subsets
```

## Reproducibility quick-start

```bash
# 1. pre-train E-ZERO (base) on LOTSA  (~GPU minutes; see results/training_logs/)
python benchmarks/pretrain.py --variant base

# 2. zero-shot evaluation + baselines
python benchmarks/run_benchmarks.py --model E-ZERO,group:naive --protocol full
python benchmarks/run_benchmarks.py --model E-ZERO,group:naive --protocol sub16

# 3. tables, figures and the publication check
python benchmarks/run_benchmarks.py --merge-only
python paper/figures/extract_tables.py
python paper/figures/regenerate_figures.py
python benchmarks/verify_publication_ready.py
```

Other entry points:

```bash
python benchmarks/run_benchmarks.py --list                                   # 19 registered models
python benchmarks/pretrain.py --variant ALL                                  # base + 7 ablations (Table 5)
python benchmarks/run_benchmarks.py --model group:ezero-ablation --protocol full
python benchmarks/run_benchmarks.py --model DLinear,PatchTST,iTransformer --protocol full
python benchmarks/run_benchmarks.py --model Chronos-Bolt-T --protocol sub16  # needs [chronos]
python benchmarks/run_efficiency.py --model E-ZERO,group:naive
python benchmarks/pretrain.py --variant base --synthetic --steps 200         # CPU smoke test
```

## Running on Kaggle

The full pipeline is split into private GPU (T4) script kernels. Each kernel clones this
repository **at a pinned commit**, installs it, runs one stage and saves its artifacts under the
repository's own paths in `/kaggle/working/outputs/`.

| stage | what it runs |
|---|---|
| `smoke` | a few minutes of every code path (check the environment before long runs) |
| `core` | pre-train E-ZERO, evaluate E-ZERO + naive baselines (both protocols, both scalings), efficiency |
| `ablations` | pre-train and evaluate the 7 ablation variants (Table 5) |
| `tsfm` | Chronos-T, Chronos-Bolt-T, TTM, Moirai-S, TimesFM-2.5 on the `sub16` protocol, one virtualenv each |
| `fullshot` | DLinear, PatchTST, iTransformer trained per dataset and horizon |

```bash
pip install kaggle                                   # needs Kaggle API credentials
git push                                             # kernels clone the pushed commit
python kaggle/build_kernels.py --user <kaggle-user> --push smoke
python kaggle/build_kernels.py --user <kaggle-user> --push core ablations
python kaggle/fetch_outputs.py --user <kaggle-user> --status
python kaggle/fetch_outputs.py --user <kaggle-user>  # copies artifacts back into the repo
python benchmarks/run_benchmarks.py --merge-only
```

Kaggle allows two concurrent GPU batch sessions per account; push stages accordingly.
The LSF CSVs come from the attached public dataset
`giochelavaipiatti/time-series-forecasts-popular-benchmark-datasets` (checked by SHA-256),
ETTh2/ETTm2 from the official ETDataset GitHub repository, and LOTSA and the foundation-model
weights from HuggingFace (internet enabled).

## Evaluation protocol

- Context length 512; horizons 96 / 192 / 336 / 720; one E-ZERO checkpoint serves all horizons
  (shorter horizons are prefixes of the 720-step forecast).
- Standard Informer / Time-Series-Library splits (ETTh 12/4/4 months, ETTm the same at 15 min,
  others 70/10/20) and every test window with stride 1 (`drop_last=False`): protocol **`full`**.
- **`sub16`**: 16 evenly spaced test windows (all channels), shared by all horizons and all
  models. Used for the head-to-head comparison with the slow foundation models (the thesis notes
  that evaluating every sliding window is impractical for them).
- Metrics: MSE and MAE on standard-scaled values. The scaler is fitted on the **train split
  only** (`--scaler train`, default). The thesis normalised with full-series statistics and
  flagged it as possible leakage (Sec. 3.6.1); `--scaler full` reproduces that choice and
  `paper/tables/tableS1_normalisation` reports both.

## What changed from the original EZERO code

The original repository could not run:
- `models/nbeats.py` was a copy of the TCN file, so `NBeatsBlock` did not exist and
  `models/hybrid_model.py` failed to import.
- `scripts/train.py` and `scripts/eval.py` imported modules that did not exist
  (`data.dataloader`, `data.preprocessing`) and read config keys that were not in the YAML files.
- Training used synthetic data rather than LOTSA, and there was no evaluation on the benchmarks.
- `checkpoints/model_weights.pth` was a 2-byte file (a line break), not a checkpoint.
- MRC, DTP, RAE, CRRB and the Exogenous Mixer described in the thesis were not implemented.

This version implements the architecture as described in the thesis (N-BEATS first, then TCN
refinement, Eq. 9-10 and 22-26). It keeps the final hyperparameters from the thesis and the
original YAML files: lr 3e-4, weight decay 1e-3, batch 64, 3 N-BEATS blocks × 64 neurons,
4 TCN layers with dilations 1/2/4/8 and 32 filters, 10k warm-up steps then cosine annealing,
gradient clipping 3.0 and dropout 0.2. It also keeps the original LOTSA sub-dataset list and
the frequency-to-patch-size map, which now sets the DTP pooling window.

## Results

See `paper/tables/` and `paper/figures/` (regenerated from `results/benchmark_final_merged.csv`).
The numbers come from the Kaggle runs recorded in `results/manifest.json` and
`results/kaggle_logs/`. They were produced with the leakage-free protocol above, so they are not
directly comparable to the thesis tables.

## Citation

See `CITATION.cff`.
