**Long-sequence forecasting MSE, full test set (stride 1): zero-shot E-ZERO vs full-shot architectures trained on each dataset**

| Dataset | E-ZERO | PatchTST | iTransformer | DLinear | SeasonalNaive | Naive | Mean |
|---|---|---|---|---|---|---|---|
| ETTh1 | _0.450_ | **0.437** | 0.458 | 0.454 | 0.600 | 1.321 | 0.708 |
| ETTh2 | **0.352** | _0.363_ | 0.398 | 0.485 | 0.483 | 0.539 | 0.391 |
| ETTm1 | 0.564 | **0.371** | _0.385_ | 0.415 | 0.489 | 1.271 | 0.696 |
| ETTm2 | **0.287** | 0.298 | _0.289_ | 0.377 | 0.358 | 0.385 | 0.346 |
| Weather | 0.271 | **0.228** | _0.242_ | 0.270 | 0.371 | 0.352 | 0.316 |
| Electricity | 0.286 | _0.167_ | **0.163** | 0.171 | 0.330 | 1.612 | 0.861 |
| Traffic | 0.759 | _0.426_ | **0.407** | 0.528 | 1.161 | 2.765 | 1.403 |
| Average | 0.424 | **0.327** | _0.334_ | 0.386 | 0.542 | 1.178 | 0.674 |
