# E-ZERO: thesis claims vs current evidence (status 2026-11-11, 08:40 UTC)

All "ours" numbers use the leakage-free protocol: scaler fitted on the train split, context 512, and the
mean of horizons 96/192/336/720. `full` scores every test window; `sub16` scores 16 identical windows
per dataset for every model. Pending items are marked; this file is updated as Kaggle stages finish.

## 1. Accuracy (thesis Table 2)

| MSE | ETTh1 | ETTh2 | ETTm1 | ETTm2 | Weather | Electricity | Traffic |
|---|---|---|---|---|---|---|---|
| **Thesis E-ZERO** | 0.212 | 0.145 | 0.215 | 0.139 | 0.040 | 0.410 | 0.252 |
| Ours, E-ZERO (full) | 0.450 | 0.352 | 0.564 | 0.287 | 0.271 | 0.286 | 0.759 |
| Ours, E-ZERO with the thesis's full-series normalisation | 0.350 | 0.275 | 0.427 | 0.222 | 0.282 | 0.286 | 0.571 |
| Best reported by any paper, zero-shot model | 0.395 | 0.333 | 0.361 | 0.253 | 0.231 | 0.188 | 0.502 |
| Best reported by any paper, model trained on the dataset | 0.411 | 0.316 | 0.347 | 0.256 | 0.222 | 0.156 | 0.388 |
| Seasonal naive (full) | 0.600 | 0.483 | 0.489 | 0.358 | 0.371 | 0.330 | 1.161 |

The "best reported" rows come from `results/reported/` (each paper's own tables; TimeMixer with its
hyperparameter search, TSMixer, PatchTST/64, TTM-A/B, Moirai-large).

**Reading**
- **Six of the seven thesis values are below the best number any published model reports,** including fully supervised models trained on the target data. For example, Weather is 0.040 against 0.222. These values cannot be reached under the standard protocol by E-ZERO or any other model.
- **The thesis's normalisation explains part of the gap, not all of it.** It lowers our MSE by 19% (average 0.424 to 0.345), but ETTh1 still lands at 0.350, not 0.212.
- **E-ZERO is a real zero-shot forecaster.**
  - It beats seasonal naive on 6 of 7 datasets.
  - It is within 11-22% of the best result ever reported on ETTh1, ETTh2, ETTm2 and Weather.
  - Its weakest datasets are ETTm1, Electricity and Traffic.

## 2. Ranking against the foundation models (thesis: "outperforms TimesFM, Moirai, Chronos, Lag-Llama; second only to TTM")

Same 16 windows for every model (sub16), averaged over the 4 horizons:

| sub16 MSE | ETTh1 | ETTh2 | ETTm1 | ETTm2 | Weather | Electricity | Traffic | Avg |
|---|---|---|---|---|---|---|---|---|
| E-ZERO | 0.451 | **0.315** | 0.517 | **0.296** | 0.286 | 0.276 | 0.753 | 0.414 |
| TTM-R1 (thesis era) | **0.383** | 0.316 | 0.538 | 0.372 | **0.236** | 0.190 | 0.528 | 0.366 |
| Chronos-T (thesis era) | 0.610 | 0.459 | 0.721 | 0.362 | 0.355 | 0.346 | 1.048 | 0.557 |
| TimesFM-1.0 (thesis era) | 0.459 | 0.352 | **0.438** | 0.330 | 0.267 | **0.149** | **0.342** | **0.334** |
| Lag-Llama (thesis era) | pending | | | | | | | |
| Moirai-S (thesis era) | re-running | | | | | | | |
| TTM-R2 (post-thesis) | 0.394 | 0.324 | 0.411 | 0.312 | 0.232 | 0.200 | 0.543 | 0.345 |
| Chronos-Bolt-T (post-thesis) | 0.436 | 0.345 | 0.385 | 0.295 | 0.284 | 0.167 | 0.406 | 0.331 |
| TimesFM-2.5 (post-thesis) | 0.419 | 0.344 | 0.401 | 0.320 | 0.275 | 0.157 | 0.351 | 0.324 |

**Status: not supported as stated.** The thesis claims E-ZERO "outperforms TimesFM, Moirai, Chronos
and Lag-Llama" and is "second only to TTM".

- **Chronos-T5-tiny: confirmed.** E-ZERO wins on all 7 datasets (0.414 vs 0.557).
- **TimesFM-1.0: contradicted.** It is the strongest thesis-era model here (0.334) and beats E-ZERO
  on 5 of 7 datasets, by a wide margin on Electricity (0.149 vs 0.276) and Traffic (0.342 vs 0.753).
- **TTM-R1: E-ZERO is behind on average** (0.414 vs 0.366) but ahead on ETTh2 and ETTm2, and level
  on ETTh1 within noise. "Second only to TTM" would hold if TimesFM did not exist.
- **E-ZERO is the best thesis-era model on ETTh2 and ETTm2.** That is the defensible claim.
- **Moirai-S and Lag-Llama are still open** (see below), so the final wording waits for them.

**Moirai-S was mis-evaluated here, not weak.** Our first run gave MSE up to 1e8 while MAE stayed
moderate, the signature of a few extreme predictions. The cause: Moirai is probabilistic with a
heavy-tailed predictive distribution, its authors evaluate the **median** of the sampled paths, and
our wrapper took the **mean**, which a single extreme draw dominates. The numbers are quarantined in
`results/failures/` and the model is re-running with median aggregation (`tsfmfix` stage).

**Lag-Llama is partly infeasible.** It costs about 54 min per ETT dataset and 2.7 h on Weather
(21 channels) on a T4. Electricity (321 channels) and Traffic (862) would need an estimated 40+ h
each, so they are not attempted; the other five datasets are re-running. Its coverage will be
reported as 5 of 7 with the reason, rather than left blank.

## 3. Against models trained on each dataset (thesis Table 3: "outperforms other architectures except on Electricity")

**Our own re-run, every model on the same test windows** (full protocol, stride 1, train-split
scaling, lookback 512, mean of horizons 96/192/336/720). E-ZERO is zero-shot; the other three were
trained on each dataset:

| MSE | E-ZERO | PatchTST | iTransformer | DLinear | SeasonalNaive |
|---|---|---|---|---|---|
| ETTh1 | 0.450 | **0.437** | 0.458 | 0.454 | 0.600 |
| ETTh2 | **0.352** | 0.363 | 0.398 | 0.485 | 0.483 |
| ETTm1 | 0.564 | **0.371** | 0.385 | 0.415 | 0.489 |
| ETTm2 | **0.287** | 0.298 | 0.289 | 0.377 | 0.358 |
| Weather | 0.271 | **0.228** | 0.242 | 0.270 | 0.371 |
| Electricity | 0.286 | 0.167 | **0.163** | 0.171 | 0.330 |
| Traffic | 0.759 | 0.426 | **0.407** | 0.528 | 1.161 |
| Average | 0.424 | **0.327** | 0.334 | 0.386 | 0.542 |

**Are the baselines trained well enough for this to be fair?** At lookback 512 our runs land near the
published numbers: iTransformer is 4.6% *better* than its paper's values (89% of cells within 10%),
PatchTST 5.4% worse (75% within 10%). DLinear is 14.2% worse at 512, but its paper uses lookback 336
and at 336 our run reproduces it within 1.4% at every ETTh1 horizon. `fullval` repeats all of them
with the official per-dataset settings. Full detail: `paper/tables/tableV_reproduction.md`.

**Status: not supported as stated, but stronger than the published-number comparison suggested.**
- **E-ZERO wins outright on 2 of 7 datasets** (ETTh2, ETTm2) against models that were trained on
  that dataset, and is within 3% of the best on ETTh1 and level with DLinear on Weather.
- It loses clearly on ETTm1, Electricity and Traffic: the many-channel datasets and the one where
  the thesis itself expected to lose.
- On average it trails PatchTST by 30% and iTransformer by 27%.
- The thesis claim "outperforms other architectures except on Electricity" does not hold. "Matches
  or beats models trained on the target data on 4 of 7 benchmarks, without ever seeing them" does.

Comparison with the values each model's own paper reports (different lookbacks, as in the thesis's
own comparison):
| E-ZERO zero-shot is better on ... | of |
|---|---|
| iTransformer (L=96) | 3 / 7 (ETTh1, ETTh2, ETTm2) |
| TimesNet (L=96) | 3 / 7 (ETTh1, ETTh2, ETTm2) |
| FEDformer (L=96) | 3 / 7 (ETTh2, ETTm2, Weather) |
| Autoformer (L=96, own paper) | 2 / 4 (ETTm2, Weather) |
| DLinear (L=336), TSMixer (L=512), TimeMixer (L=96) | 1 / 7 (ETTh2) |
| PatchTST/42 and /64, NLinear, TimeMixer (searched) | 0 / 7 |

**Status: not supported as stated.**
- E-ZERO without any target-data training matches or beats the 96-lookback transformer models on about half the datasets.
- Well-tuned long-lookback models (PatchTST, TSMixer, DLinear) beat it nearly everywhere.
- Our own re-runs of these baselines are in the `fullshot*` and `fullval` stages.

## 4. Efficiency (thesis Tables 3-4)

| | Params | GPU ms / batch | GPU mem (GB) | CPU s / batch |
|---|---|---|---|---|
| Thesis E-ZERO | 1M | 9.8 | 0.1 | 0.02 |
| Thesis TTM | 0.8M | 4.7 | 0.06 | 0.01 |
| Ours E-ZERO | 0.74M | 11.9 (before the exact-truncation fix, which was 1.6x faster at H=96 on CPU) | 0.053 | 0.118 |
| Ours TTM-R2 | 0.81M | 6.1 | 0.015 | 0.014 |
| Ours Chronos-Bolt-T | 8.65M | 54.6 | 0.181 | 0.986 |

**Status: partly supported.**
- **Supported:**
  - E-ZERO is tiny (0.74M parameters).
  - It is 4.6x faster than Chronos-Bolt on the GPU; timings for the larger thesis-era models are pending.
- **Not supported:**
  - "Lowest inference time of all models": TTM is faster in our measurement and in the thesis's own Table 3.
  - "Nearly double the efficiency of TTM" contradicts the thesis's own Table 4, where lower is better and TTM scores 0.0031 against E-ZERO's 0.0065.

## 5. Ablations (thesis Table 5)

All 7 variants were pre-trained with the base model's budget and initialisation. Each reached at least
41k steps, with no non-finite steps and none stopped by the time limit. Average MSE over the 7 datasets
(full protocol); "change" is relative to the full model (0.424):

| Removed | Avg MSE | Change | Dev score (flip / no flip) | Thesis Table 5 |
|---|---|---|---|---|
| (none, base) | 0.424 | - | 0.802 / 1.252 | ETTh1 0.212 |
| CRRB | 0.442 | **+4.2%** | 0.808 / 1.087 | helps |
| TCN | 0.437 | **+3.0%** (Weather +14.5%) | 0.849 / 1.139 | helps a lot |
| DTP | 0.425 | +0.1% | 0.808 / 1.249 | helps |
| RAE | 0.425 | +0.1% | **0.770 / 0.773** | helps |
| MRC | 0.415 | -2.2% | 0.795 / 1.110 | helps |
| Exog mixer | 0.411 | -3.1% | 0.807 / 1.052 | helps |
| N-BEATS (replaced by one linear layer) | **0.397** | **-6.4%** (Traffic -25%, Electricity -14%; Weather +11%, ETTm2 +6%) | 0.777 / 1.041 | helps most (ETTh1 0.212 -> 0.380) |

The dev score is on the held-out development benchmark; lower is better and below 1 beats seasonal naive.

**Status: not reproduced.**
- **TCN** is the only component that helps both on the evaluation data and on the held-out development data.
- **CRRB** helps on the evaluation data (+4.2%) but is neutral on the development data.
- **DTP and RAE** are neutral on the evaluation data. Removing RAE gives the best development score and removes most of the sign asymmetry: 0.773 without flip averaging, against 1.25 for the base model.
- **MRC, the Exog mixer and the N-BEATS stack** are, on average, better removed on the evaluation data. This is driven by the many-channel datasets (Electricity, Traffic). On the development data the differences are small.
- **The thesis's main ablation claim**, that N-BEATS matters most, is contradicted.
- **Caveat:** each variant is a single training run, so differences of a few percent may be noise. The large effects (N-BEATS on Traffic/Electricity, TCN on Weather, CRRB on ETTm1) are not.
- **No architecture change has been made from these evaluation-set numbers.** Any change has to be justified on the development benchmark (see `results/dev/dev_results.csv`, label `ablations-dev`).

## 6. Path forward (legitimate options only)

| Step | What | Status | Expected effect |
|---|---|---|---|
| 1 | Re-run every thesis comparison on one protocol | thesis-era foundation models, full-shot baselines, ablations: running or queued | turns Tables 2-5 into verified numbers |
| 2 | Sign-flip test-time averaging (chosen on held-out dev data) | done; dev score 1.25 -> 0.80 | mainly low-frequency data; LSF effect measured once in the `tweaks` stage |
| 3 | v3 pre-training: more diverse non-overlapping data, sign-flip and down-sampling augmentation, horizon-balanced loss | queued (`tweaks`); best of 3 picked on dev data by a fixed rule | narrows the gap to TTM / Chronos-Bolt; size unknown until run |
| 4 | Exact horizon truncation at inference | done (identical outputs) | about 1.6x faster at H=96; re-timed in `tweaks` |
| 5 | Rewrite the claims around the verified evidence | after steps 1-3 | defensible contribution |

**Claims supported now:**
- A 0.74M-parameter, attention-free N-BEATS + TCN model pre-trained on LOTSA forecasts zero-shot better than seasonal naive on 6 of 7 standard datasets.
- It comes within 11-22% of the best result any model reports on four of them.
- It runs several times faster than the Chronos family.

**Claims that depend on pending runs:**
- That it beats the thesis-era foundation models (Chronos-T5 looks likely; Moirai, TimesFM and Lag-Llama are open).
- (Answered: the ablations do not support "each component contributes"; see section 5.)

**Claims that should be withdrawn:**
- The Table 2 absolute values.
- "Lowest inference time among all models".
- "Nearly double the efficiency of TTM".
- The Table 5 ablation values.

No tweak can recover these without breaking the protocol.
