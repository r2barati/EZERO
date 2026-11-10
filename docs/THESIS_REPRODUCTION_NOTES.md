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

## 5. Making the contribution claim

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
