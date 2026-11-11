**Inference time per batch (32 series, context 512, horizon 96) and memory**

| Model | GPU TIME (ms) | Params (M) | MEM (GB) | CPU TIME (s) |
|---|---|---|---|---|
| Naive | 0.0120 | - | 0.0096 | 0.0000 |
| Mean | 0.0350 | - | 0.0096 | 0.0000 |
| SeasonalNaive | 0.0240 | - | 0.0096 | 0.0000 |
| TTM-R2 | 6.0930 | 0.8053 | 0.0154 | 0.0137 |
| E-ZERO-noTCN | 6.0290 | 0.7128 | 0.0201 | - |
| E-ZERO-noNBEATS | 6.3230 | 0.4053 | 0.0509 | - |
| E-ZERO-noDTP | 9.9030 | 0.7391 | 0.0414 | - |
| E-ZERO-noCRRB | 7.3440 | 0.4079 | 0.0514 | - |
| E-ZERO | 11.9070 | 0.7401 | 0.0525 | 0.1183 |
| E-ZERO-noExog | 8.5190 | 0.7373 | 0.0529 | - |
| E-ZERO-noRAE | 9.8760 | 0.7347 | 0.0507 | - |
| E-ZERO-noMRC | 11.0310 | 0.7401 | 0.0525 | - |
| Chronos-Bolt-T | 54.5680 | 8.6527 | 0.1807 | 0.9856 |
