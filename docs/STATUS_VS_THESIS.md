# E-ZERO: thesis claims vs current evidence (status 2026-11-10, 22:00 UTC)

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
| E-ZERO | 0.451 | **0.315** | 0.517 | 0.296 | 0.286 | 0.276 | 0.753 | 0.414 |
| Chronos-T (thesis era), preliminary* | 0.587 | 0.467 | 0.709 | 0.359 | pending | pending | pending | - |
| TTM-R1, Moirai-S, TimesFM-1.0, Lag-Llama (thesis era) | pending | | | | | | | |
| TTM-R2 (released after the thesis) | 0.394 | 0.324 | 0.411 | 0.312 | 0.232 | 0.200 | 0.543 | 0.345 |
| Chronos-Bolt-T (released after the thesis) | 0.436 | 0.345 | 0.385 | 0.295 | 0.284 | 0.167 | 0.406 | 0.331 |

\*First Kaggle run, 20 samples, stopped after the ETT datasets (out of GPU memory). The final run
(10 samples, all datasets) is in progress.

**Status: partly supported.**
- **Against Chronos-T5-tiny:** E-ZERO is better on all four ETT datasets in the preliminary run, consistent with the thesis's ranking.
- **Against the other thesis-era models:** pending.
- **Against later releases:** E-ZERO trails TTM-R2 and Chronos-Bolt on average (0.414 vs 0.345 and 0.331) and is best only on ETTh2.

## 3. Against models trained on each dataset (thesis Table 3: "outperforms other architectures except on Electricity")

Comparison with the values each model's own paper reports (different lookbacks, as in the thesis's own comparison):

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

**Status: pending.**
- The thesis's ablation values could not have come from the original code, which did not implement the components.
- All 7 ablation variants are being pre-trained and evaluated now, with the same budget and initialisation as the base model.

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
- That each multi-resolution component contributes (the ablations).

**Claims that should be withdrawn:**
- The Table 2 absolute values.
- "Lowest inference time among all models".
- "Nearly double the efficiency of TTM".
- The Table 5 ablation values.

No tweak can recover these without breaking the protocol.
