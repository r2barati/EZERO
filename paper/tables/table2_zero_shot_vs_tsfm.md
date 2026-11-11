**Zero-shot MSE vs TS pre-trained models available at the time of the thesis, averaged over horizons {96, 192, 336, 720} (protocol sub16: identical test windows for every model, train-split scaling)**

| Dataset | E-ZERO | TTM-R1 | Chronos-T | TimesFM-1.0 | SeasonalNaive | Naive |
|---|---|---|---|---|---|---|
| ETTh1 | _0.451_ | **0.383** | 0.610 | 0.459 | 0.574 | 0.894 |
| ETTh2 | **0.315** | _0.316_ | 0.459 | 0.352 | 0.429 | 0.462 |
| ETTm1 | _0.517_ | 0.538 | 0.721 | **0.438** | 0.539 | 1.356 |
| ETTm2 | **0.296** | 0.372 | 0.362 | _0.330_ | 0.391 | 0.399 |
| Weather | 0.286 | **0.236** | 0.355 | _0.267_ | 0.359 | 0.345 |
| Electricity | 0.276 | _0.190_ | 0.346 | **0.149** | 0.296 | 1.529 |
| Traffic | 0.753 | _0.528_ | 1.048 | **0.342** | 1.177 | 3.156 |
| Average | 0.414 | _0.366_ | 0.557 | **0.334** | 0.538 | 1.163 |
