**Inference time per batch (32 series, context 512, horizon 96) and memory**

| Model | GPU TIME (ms) | Params (M) | MEM (GB) | CPU TIME (s) |
|---|---|---|---|---|
| Naive | 0.0120 | - | 0.0096 | 0.0000 |
| Mean | 0.0350 | - | 0.0096 | 0.0000 |
| SeasonalNaive | 0.0240 | - | 0.0096 | 0.0000 |
| TTM-R2 | 6.0930 | 0.8053 | 0.0154 | 0.0137 |
| E-ZERO | 11.9070 | 0.7401 | 0.0525 | 0.1183 |
| Chronos-Bolt-T | 54.5680 | 8.6527 | 0.1807 | 0.9856 |
