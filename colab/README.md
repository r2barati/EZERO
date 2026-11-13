# Running the remaining measurements on Colab

The Kaggle kernels in `../kaggle/` run on a weekly GPU quota. Everything they do can be run
anywhere else with a GPU, from the same code at the same commit, with:

```
python colab/run_stage.py --stage <stage> --out <a folder that outlives the session>
```

`run_stage.py` renders `kaggle/kernel_runner.py` with this checkout's remote and commit and
runs it, so the commands, the pinned commit and the output layout are identical to a Kaggle
run. Results merge with results from any other machine: every row is keyed by model, dataset,
horizon, protocol, scaler and context length.

`EZERO_remaining_runs.ipynb` is the same thing as a notebook, with the Drive mounting and the
dataset download written out. Open it with:

<https://colab.research.google.com/github/r2barati/ezero/blob/ezero-v2-paper1-structure-kaggle/colab/EZERO_remaining_runs.ipynb>

## The stages that are still outstanding

| Stage | What it measures | Roughly, on one T4 |
|---|---|---|
| `tsfmfix` | Moirai-S re-run after the sample-aggregation fix (median, as its authors evaluate); Lag-Llama on the five datasets it can finish | 1 h + 6 h |
| `tsfmval` | the foundation models under the protocols their own papers used, as a reproduction check | ~4 h |
| `fullval` | the full-shot baselines with their published per-dataset settings, 3 seeds on ETTh1 | ~6 h |
| `fullshot2` | TSMixer, TimeMixer, TimesNet trained per dataset and horizon | ~5 h |
| `fullshot3` | FEDformer, Autoformer, the same way | ~6 h |

`fullshot2` and `fullshot3` are what Figure 8 of the corrected thesis is missing: its footnote
names these five models as implemented and still being measured.

## Interruptions

Nothing is lost when a session ends. Artifacts are copied to `--out` after every command, and
the next run copies them back into the checkout before starting. Benchmarks are cached per
model and dataset, so the second run skips what the first finished. Re-run the same command
until the stage reports no work left.

`--hours` sets when the runner stops starting new commands (default 11, which suits Kaggle's
12 h cap). On a Colab session that may be reclaimed sooner, set it lower.

## Data

ETTh1, ETTh2, ETTm1 and ETTm2 download themselves. Weather, Electricity and Traffic come from
a Kaggle dataset and need a Kaggle API token, or the three CSVs placed in `data/raw/` by hand.
`EZERO_DATA_DIR` points the loader somewhere else if you keep them elsewhere.

## Sending results back

The `--out` folder is the whole deliverable. Its `working/outputs/` holds `results/cache/`
(one CSV per model), `results/training_logs/`, `results/dev/` and any trained weights;
`working/logs/` holds the console output of each command and the pip freeze of each
environment. Dropping `working/outputs/` into a checkout and running
`python benchmarks/run_benchmarks.py --merge-only` rebuilds the merged table.
