import numpy as np

def normalize_series(series):
    """
    Instance-level normalization function.
    Given an input series (e.g., a window of time steps), returns the normalized
    series, and its mean and standard deviation.
    """
    mean = np.mean(series)
    std = np.std(series)

    # Avoid division by zero
    if std == 0:
        std = 1.0

    normalized_series = (series - mean) / std
    return normalized_series, mean, std

def create_rolling_windows(series, input_window, forecast_horizon):
    """
    Create rolling windows for training/evaluation and apply instance-level
    normalization (RevIN-style) independently for each window.

    Optimized for performance with numpy array slicing.
    """
    if len(series) < input_window + forecast_horizon:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Vectorized windowing
    num_windows = len(series) - input_window - forecast_horizon + 1

    # Create an array of indices for X and Y
    idx_x = np.arange(input_window)[None, :] + np.arange(num_windows)[:, None]
    idx_y = np.arange(input_window, input_window + forecast_horizon)[None, :] + np.arange(num_windows)[:, None]

    X_raw = series[idx_x]
    y_raw = series[idx_y]

    means = np.mean(X_raw, axis=1, keepdims=True)
    stds = np.std(X_raw, axis=1, keepdims=True)
    stds[stds == 0] = 1.0  # Avoid division by zero

    X_norm = (X_raw - means) / stds
    y_norm = (y_raw - means) / stds  # Target normalized with input stats

    return X_norm, y_norm, means.flatten(), stds.flatten()
