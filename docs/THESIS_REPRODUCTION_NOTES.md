# Thesis reproduction notes

These notes record how this repository maps to the thesis, where the thesis text was
ambiguous or did not match the original code, and which choices were made. All choices were
fixed **before** looking at any test-set result, so the numbers in `paper/tables/` are an
independent re-run, not a re-tuning toward the thesis tables.

## 1. State of the original repository

| Item | Original repository | This version |
|---|---|---|
| N-BEATS block | missing (`models/nbeats.py` was a copy of `tcn.py`) | `ezero/models/nbeats.py` |
| Hybrid model | TCN → last time step → N-BEATS (`hybrid_model.py`, could not be imported) | N-BEATS → TCN refinement, as thesis Eq. 9-10 |
| Train / eval scripts | imported missing modules, used synthetic data | `benchmarks/pretrain.py`, `benchmarks/run_benchmarks.py` |
| Pre-training data | LOTSA subsets listed in YAML, never loaded | same 10 subsets, loaded from HuggingFace |
| Checkpoint | `checkpoints/model_weights.pth` = 2 bytes (`\r\n`) | `data/models/ezero_<variant>.pt` + SHA-256 in the manifest |
| MRC / DTP / RAE / CRRB / Exog mixer | not implemented | implemented (Sec. 3.5) with ablation switches |

Because no trained weights or result logs survived, none of the thesis tables can be
re-derived from the original artifacts. The results here come from a fresh pre-training run.

## 2. Architecture decisions where the thesis is ambiguous

- **Order of components.** Sec. 3.1 and Eq. 9-10, 12-13 and 24-25 put N-BEATS first and TCN
  second. The original code did the reverse. This version follows the thesis text.
- **Exogenous Mixer in zero-shot use.** Eq. 22-26 mix channels with a TCN. A fixed-width
  channel convolution cannot take an unseen number of channels (7 in ETT, 862 in Traffic), so
  each channel sees its own forecast plus its group's mean and standard deviation. This is
  permutation invariant and works for any number of channels. Known future values of exogenous
  channels are substituted as in Eq. 22. The LSF benchmarks have no known-future covariates,
  so in these benchmarks the block only mixes target channels.
- **Not implemented** (described only in prose, with no equation or hyperparameter):
  - the attention re-weighting of features inside N-BEATS (Sec. 3.3.3), which also conflicts
    with the "non-attention" claim;
  - freezing the lower layers during pre-training (Sec. 3.3.5);
  - the embedding "similarity matching layer" and adaptive inference (Sec. 3.3.5);
  - conditional convolutions: the frequency embedding (RAE) is fed to the TCN as constant
    input channels instead.
- **Zero-variance groups in the Exogenous Mixer.** Channels of one group can forecast identical
  values, giving a group variance of exactly 0, where the gradient of `sqrt(var)` is unbounded.
  On a Kaggle T4 (torch 2.11, CUDA 13) this produced non-finite gradients in all 143 parameter
  tensors on the very first step (`scripts/debug/nan_probe.py --mixer-eps 0`), while the same
  code trained normally on CPU. The group standard deviation is therefore `sqrt(var + 1e-4)`.
  With that change the probe ran 400 finite steps; the minimum group variance seen was 0 to
  about 4e-7. As a backstop, training also skips any optimiser step with a non-finite loss or
  gradient norm and records the count.
- **One head for all horizons.** The model always predicts 720 steps; 96/192/336 are prefixes.
- **Loss robustness.** Pre-training uses MSE in the instance-normalised space. Targets are
  clipped to ±10 context standard deviations and windows with a constant context are skipped.
  Without this, a handful of LOTSA windows (sensor series that sit at zero, then jump) dominate
  the loss.

## 3. Protocol corrections

- **Normalisation leakage.** Sec. 3.6.1 states that the evaluation data were normalised with
  statistics of the whole series, including validation and test, and flags this as leakage.
  The default here (`--scaler train`) fits the scaler on the train split only, which is the
  standard Informer / Time-Series-Library practice. `--scaler full` reproduces the thesis
  choice as a diagnostic (`paper/tables/tableS1_normalisation`).
- **Instance normalisation** inside E-ZERO uses only observed context values, so it cannot
  leak target information either.
- **Test windows.** All windows with stride 1 and `drop_last=False` (protocol `full`). The slow
  foundation models are compared on 16 evenly spaced windows (protocol `sub16`), and E-ZERO is
  reported on the same windows, so the comparison is like-for-like.
- **No overlap between pre-training and evaluation data.** `load_corpus` refuses LOTSA subsets
  that share a source with the evaluation sets (e.g. `traffic_hourly` is PeMS-SF, the source of
  the Traffic benchmark).
- **Model selection** uses only the held-out tail of the LOTSA series (early stopping). No
  hyperparameter was chosen using LSF validation or test data.

## 4. Thesis numbers to treat with care

- **Table 2 mixes two protocols.** The Moirai-S column (ETTh1 0.400, ETTh2 0.341, ETTm1 0.448,
  ETTm2 0.300, Weather 0.242, Electricity 0.233) appears to reproduce the values published in
  the Moirai paper, which uses train-split normalisation. The E-ZERO column was computed with
  the full-series normalisation of Sec. 3.6.1. Numbers from the two protocols are not on the
  same scale, so the ranking in Table 2 is not established by the thesis itself. Re-running
  every model on the same windows with the same scaler (`sub16`) is what this repository adds.
- **Table 5 (ablation) has no code behind it.** It reports results for components that did not
  exist in the code, and its increments are very regular.
- **Table 3 (inference cost) has some implausible values,** e.g. Chronos-T CPU time 2504 s per
  batch next to a GPU time of 1.4 s. `benchmarks/run_efficiency.py` re-measures every model on
  one fixed workload.

## 5. Baseline versions: thesis era vs later releases

The thesis (August/September 2024) compared against foundation models that were public at the
time. Several have newer releases since; using them would compare E-ZERO against models the thesis
could not have known. The registry therefore tags every foundation model with an era (HuggingFace
creation date of the checkpoint):

| Thesis column | Thesis-era checkpoint used here | Later release (context only) |
|---|---|---|
| TTM | `granite-timeseries-ttm-r1` 512-96, 2024-04 (rolled to longer horizons) | `ttm-r2`, 2024-10 |
| Chronos (ChronosT) | `chronos-t5-tiny`, 2024-02 | `chronos-bolt-tiny`, 2024-11 |
| Lag-Llama | `Lag-Llama`, 2024-02 | - |
| Moirai-S | `moirai-1.0-R-small`, 2024-02 | - |
| TimesFM | `timesfm-1.0-200m-pytorch`, 2024-08 | `timesfm-2.5-200m`, 2025-09 |

Table 2 (`paper/tables/table2_zero_shot_vs_tsfm`) uses only the thesis-era checkpoints.
`table2c_post_thesis_models` reports the later releases on the same windows, as context.
The full-shot architectures (DLinear 2022, PatchTST 2023, iTransformer 2024) all predate the thesis.
Not re-run: MOMENT and GPT4TS (Table 3), which need a trained forecasting head rather than
zero-shot inference, and TSMixer, TimeMixer, TimesNet, FEDformer and Autoformer.

Version notes: TTM-R1's public checkpoint has a 96-step head, so horizons 192-720 are produced by
rolling it forward on its own forecasts. Chronos-T uses 10 samples instead of the default 20,
because 20 x 256 series ran out of GPU memory and Traffic would have taken hours. TimesFM-1.0 and
Lag-Llama run in a Python 3.11 environment because their libraries do not support Python 3.13.
Lag-Llama is opt-in: it is not a pip package, and running it means running the authors' GitHub
code (pinned commit `df7531a`). It runs only when named explicitly in the `tsfm` stage.

## 6. Revising the thesis safely

The accepted thesis is an archived document and should stay as it is. Corrections belong in a
separate, clearly dated record that cites it:

1. Discuss the findings with the supervisor and ask the graduate school / library how they handle
   an erratum or addendum to a deposited thesis. Procedures vary; do not edit the deposited PDF.
2. Publish the corrected results (this repository, plus an arXiv preprint or journal paper)
   stating plainly what changed and why: the protocol (train-split scaling instead of full-series
   normalisation), re-run baselines instead of copied numbers, the implemented components, and the
   actual ablation results.
3. Report the re-run numbers whatever they are. If E-ZERO does not beat the larger models on
   accuracy, the claim becomes an accuracy-per-cost statement rather than "outperforms".
4. Keep the thesis-normalisation numbers (`tableS1_normalisation`) as a diagnostic that explains
   the gap to the thesis tables, not as headline results.

Never adjust the model, data or evaluation to move the test numbers toward the thesis tables. That
includes choosing settings by test score, reporting the leaky normalisation as the main result,
dropping datasets or horizons, under-training baselines, or mixing copied and re-run numbers.

## 7. Reproduction check against the numbers each model's paper reports

**Sources.** `results/reported/*.csv` holds the long-sequence results reported by each baseline's
own paper, or by the closest related paper where the model's own paper has none. Every value was
parsed from the arXiv HTML tables, cross-checked against the PDFs, and stored with paper, table,
lookback and URL. `paper/figures/compare_reported.py` joins each reported cell to our run of the
same model, dataset, horizon, lookback and test protocol, and writes
`paper/tables/tableV_reproduction.md` plus the row-level `tableV_reproduction_rows.csv`.

**How the protocols differ, and how the check handles it**

| Model | Reported protocol | Our canonical run | Like-for-like validation run |
|---|---|---|---|
| DLinear | lookback 336, single run | 512, budget cap | `fullval`: 336, official settings, 3 seeds ETTh1 + all datasets |
| PatchTST | /64: lookback 512; /42: 336; reduced model on ETTh1/2; test loader `drop_last=True` | 512 | `fullval`: /42 official scripts (336, type3/OneCycle, 100 epochs) |
| TSMixer | lookback 512, hyperparameters tuned per dataset and horizon (Table 8), 100 epochs | 512, fixed widths | `fullval`: Table 8 settings, authors' block design |
| iTransformer, TimesNet, TimeMixer, FEDformer, Autoformer | lookback 96, official widths (FEDformer/Autoformer d_model 512, no instance norm), calendar features | 512, reduced widths | `fullval`: 96, official run-script settings |
| TTM | per-horizon checkpoints, every sliding window (TTM Table 11) | R1: one 96-step model rolled forward | `tsfmval`: full test set |
| Moirai | context 1000-5000, patch size and CI/channel-mixing tuned on validation, median of samples | context 512, patch 32, mean of 20 samples | `tsfmval`: full test set (our fixed settings) |
| TimesFM | TTM Table 13 (sliding) and its own Table 5 (last window, MAE) | sub16 | `tsfmval`: full + last window |
| Chronos, Lag-Llama | no own LSF numbers; TTM Table 15 re-evaluation, last window only; Lag-Llama pretrained on ETTh1/ETTh2/ETTm1/Electricity/Traffic | sub16 | `tsfmval`: last window |

**Findings so far**

- **DLinear at its paper lookback (336) reproduces.**
  - ETTh1: within 1.4% at every horizon (0.377 vs 0.375 at H=96).
  - ETTh2: within 2.5% at H=96/192.
  - ETTh2 at H=336/720 differs by -13% and +22%. Five seeds give 0.389-0.451 (reported 0.448) and 0.608-0.740 (reported 0.605), so the reported single runs sit at the edge of normal seed variance. The cause is seed variance, not a bug.
  - Emulating the paper-era test loader (`drop_last=True`, protocol `fulldl32`) changes MSE by less than 1.2%, so it does not explain the gaps.
- **A missing LR schedule was the one real bug found.**
  - At the reference learning rate 5e-3 and without the per-epoch halving of the reference code, DLinear scored 0.463 on ETTh1 H=96. The full-shot trainer now applies the reference `type1` schedule exactly, which gives 0.372-0.374 (reported 0.375).
- **Not reproducible by design:**
  - Calendar ("time-mark") embeddings, used by iTransformer, TimesNet, FEDformer and Autoformer, are not implemented here.
  - Moirai's validation-tuned context is not replicated.
  - TTM's per-horizon checkpoints are not used: the public R1 checkpoint is rolled forward instead.
  - These are recorded as residual protocol differences rather than tuned away.

## 8. Making the contribution claim

The claim should be stated from the re-run numbers, whichever way they fall. The thesis'
central argument is efficiency (an attention-free model at about 1M parameters, with
competitive zero-shot accuracy), and that is directly testable here:

1. accuracy vs naive and seasonal-naive baselines (a zero-shot model must clearly beat these);
2. accuracy vs the foundation models on identical windows (`sub16`);
3. accuracy vs full-shot models trained on each target dataset (`full`);
4. parameters, latency and memory (`results/efficiency.csv`);
5. which components actually help (`paper/tables/table5_ablation`).

If E-ZERO does not beat the larger foundation models on accuracy, the defensible framing is an
accuracy-per-cost trade-off (e.g. error vs latency/parameters), not "outperforms".
Legitimate ways to improve the model further are a larger, still non-overlapping LOTSA corpus,
longer pre-training, and hyperparameter search on the LOTSA validation tail. Test-set-driven
tuning, leaky normalisation, or under-trained baselines would invalidate the comparison.

## 9. Improving E-ZERO without breaking the zero-shot claim

**Rule.** Every tweak is chosen on a development benchmark that shares no data with the evaluation
datasets: no ETT, Weather, Electricity or Traffic, not even their validation splits. The benchmark
(`ezero/devsuite.py`) has 7 held-out LOTSA datasets:

- kdd_cup_2018 (hourly)
- pedestrian_counts (hourly)
- elecdemand (30-minute)
- covid19_energy (hourly)
- saugeenday (daily)
- us_births (daily)
- sunspot (daily)

None is in any pre-training corpus here; `load_corpus` refuses them. The score is the geometric mean over datasets of MSE / MSE(seasonal naive), so lower is better and below 1 beats seasonal naive. The selection rule (`benchmarks/select_v3.py`) is fixed before any evaluation result of the candidates exists. The selected model is evaluated once, and every candidate's development score is published in `results/dev/`.

**Development results for the thesis-configuration checkpoint**

| Candidate | Dev score | kdd | pedestrian | elecdemand | covid | saugeen | births | sunspot |
|---|---|---|---|---|---|---|---|---|
| E-ZERO | 1.252 | 0.536 | 0.813 | 0.634 | 0.928 | 1.403 | 8.879 | 1.513 |
| E-ZERO + flip averaging | **0.802** | 0.529 | 0.798 | 0.653 | 0.903 | 0.599 | 2.641 | 0.540 |
| E-ZERO + multi-context averaging | 1.514 | 0.552 | 0.905 | 0.805 | 0.984 | 2.003 | 13.25 | 1.736 |
| Mean (reference) | 1.055 | | | | | | | |

**What this shows.** E-ZERO is good on sub-daily data and poor on daily data. Averaging f(x) with -f(-x) fixes most of the daily failures, which points to a sign asymmetry and too little low-frequency data in pre-training (the corpus has only two small daily datasets).

**v3 candidates.** All keep the thesis model configuration (0.74M parameters) and optimiser:

| Candidate | Change |
|---|---|
| v3a | extended non-overlapping corpus: 41 LOTSA subsets including M4, NN5, tourism, FRED-MD and web traffic (`configs/pretrain_corpus_extended.yaml`) |
| v3b | v3a + random sign-flip augmentation (p = 0.5) |
| v3c | v3b + down-sampling augmentation (p = 0.3, factors 2 and 4) + a horizon-balanced loss in which each of the four evaluation horizons weighs equally |

Each candidate is scored on the development benchmark with and without flip averaging, together with the thesis-configuration checkpoint. The lowest score wins (ties go to the simpler option) and becomes `E-ZERO-v3`. `E-ZERO-flip` (the thesis checkpoint with flip averaging) is reported as well. All candidates and their scores are published, including any that do not help.

**What the tweaks cannot do.** No legitimate change brings ETTh1 to the thesis value of 0.212. Under the standard protocol the strongest published models report about 0.36-0.40 on ETTh1, and our TTM and Chronos-Bolt re-runs are in that range. The realistic goal is to narrow the gap to the strongest zero-shot baselines at E-ZERO's size and cost.
