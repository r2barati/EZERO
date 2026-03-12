import numpy as np
import pandas as pd

def generate_time_series(num_series, min_length, max_length):
    """
    Generate synthetic time series data to simulate zero-shot and few-shot scenarios
    with varying properties such as length, scale, trend, seasonality, and noise.
    """
    data = {}

    for i in range(num_series):
        # Randomly choose length
        length = np.random.randint(min_length, max_length + 1)

        # Time array
        t = np.arange(length)

        # Random scale
        scale = np.random.uniform(1.0, 1000.0)

        # Random trend
        trend = np.random.uniform(-0.05, 0.05) * t

        # Random seasonality
        seasonality_period = np.random.randint(7, 365)
        seasonality = np.sin(2 * np.pi * t / seasonality_period) * np.random.uniform(0.5, 5.0)

        # Random noise
        noise = np.random.normal(0, np.random.uniform(0.1, 2.0), length)

        # Combine components
        series = scale + trend + seasonality + noise

        # Pad with NaN to match max_length so we can store in a DataFrame easily
        # or just return as a dictionary / dataframe of lists
        padded_series = np.pad(series, (0, max_length - length), constant_values=np.nan)
        data[f'series_{i}'] = padded_series

    return pd.DataFrame(data)
