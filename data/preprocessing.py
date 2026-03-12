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

    Returns:
    - X: Array of normalized input windows
    - y: Array of corresponding normalized target windows
    - means: Array of means for each input window
    - stds: Array of standard deviations for each input window
    """
    if len(series) < input_window + forecast_horizon:
        return np.array([]), np.array([]), np.array([]), np.array([])

    X, y, means, stds = [], [], [], []
    for i in range(len(series) - input_window - forecast_horizon + 1):
        # Extract the input window
        input_seq = series[i:i + input_window]

        # Extract the target window
        target_seq = series[i + input_window:i + input_window + forecast_horizon]

        # Normalize based ONLY on the input window statistics (preventing data leakage)
        mean = np.mean(input_seq)
        std = np.std(input_seq)

        if std == 0:
            std = 1.0

        normalized_input = (input_seq - mean) / std
        normalized_target = (target_seq - mean) / std # Target is normalized with input stats

        X.append(normalized_input)
        y.append(normalized_target)
        means.append(mean)
        stds.append(std)

    return np.array(X), np.array(y), np.array(means), np.array(stds)
