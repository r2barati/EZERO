# data/

- `raw/` (git-ignored): LSF evaluation CSVs. `python scripts/data/download_lsf.py` fills it
  (ETT from the official ETDataset GitHub repo; Weather / Electricity / Traffic from the public
  Kaggle dataset `giochelavaipiatti/time-series-forecasts-popular-benchmark-datasets`, checked
  against the SHA-256 values in `ezero/data/ltsf.py`). On Kaggle the attached dataset under
  `/kaggle/input/` is found automatically.
- `lotsa/` (git-ignored): LOTSA pre-training subsets, downloaded on demand by
  `benchmarks/pretrain.py` (or `python scripts/data/download_lotsa.py`).
- `models/`: E-ZERO checkpoints `ezero_<variant>.pt` (config + weights + training summary),
  produced by `benchmarks/pretrain.py`; SHA-256 values are recorded in `results/manifest.json`.
