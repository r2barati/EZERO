**E-ZERO MSE with leakage-free train-split scaling vs the thesis' full-series scaling**

| Dataset | E-ZERO (scaler=train) | E-ZERO (scaler=full) |
|---|---|---|
| ETTh1 | 0.450 | 0.350 |
| ETTh2 | 0.352 | 0.275 |
| ETTm1 | 0.564 | 0.427 |
| ETTm2 | 0.287 | 0.222 |
| Weather | 0.271 | 0.282 |
| Electricity | 0.286 | 0.286 |
| Traffic | 0.759 | 0.571 |
| Average | 0.424 | 0.345 |

The thesis (Sec. 3.6.1) normalised with statistics of the whole series and flagged it as possible leakage; the repository default is train-split scaling.
