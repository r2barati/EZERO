import numpy as np
import pandas as pd

def normalize_series(series):
    mean = series.mean()
    std = series.std()
    return (series - mean) / std, mean, std

def create_rolling_windows(series, input_window, forecast_horizon):
    if len(series) < input_window + forecast_horizon:
        return np.array([]), np.array([])

    X, y = [], []
    for i in range(len(series) - input_window - forecast_horizon + 1):
        X.append(series[i:i + input_window])
        y.append(series[i + input_window:i + input_window + forecast_horizon])
    return np.array(X), np.array(y)
