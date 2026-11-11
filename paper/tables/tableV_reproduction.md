**Reproduction check: our MSE vs the MSE reported in each model's paper** (gap = ours / reported - 1; positive = we are worse)

| model | paper variant | comparison | seeds | cells | median gap % | max abs gap % | within 5% | within 10% | reproduced* |
|---|---|---|---|---|---|---|---|---|---|
| DLinear | DLinear / Zeng et al. AAAI 2023 Table 2 | L512 | 1 | 28 | +14.2 | 67.4 | 29% | 36% | 29% |
| PatchTST | PatchTST/42 / Nie et al. ICLR 2023 Table 3 | L512 | 1 | 28 | +5.4 | 21.3 | 43% | 75% | 43% |
| iTransformer | official / Liu et al. ICLR 2024 Table 10 (App. F.4) | L512 | 1 | 28 | -4.6 | 12.3 | 46% | 89% | 46% |
| PatchTST | PatchTST/64 / Nie et al. ICLR 2023 Table 3 | matched (lookback only) | 1 | 28 | +6.2 | 22.9 | 36% | 75% | 36% |

*reproduced = gap within 5%, or the reported value lies inside the range of our seeds.

`matched` = run at the paper's lookback; `L512` = only our canonical 512-step run exists, so the gap mixes protocol and implementation differences. Row-level data: tableV_reproduction_rows.csv.
