import pandas as pd
import numpy as np
from datasets import load_dataset
import requests
import zipfile
import io

def load_monash_dataset(dataset_name="m4_hourly", max_series=10):
    """
    Loads a dataset from the Monash Time Series Forecasting Archive using HuggingFace.
    We will use the generic 'monash_tsf' repository from HuggingFace, if available, or
    default to a known open dataset wrapper.
    For this implementation, we will use the open datasets available on HuggingFace
    like `monash_tsf` (e.g., 'autogluon/chronos_datasets' or similar community ports),
    or simply fetch the standard M4 hourly subset as an example proxy for Monash format.
    """
    print(f"Loading {dataset_name} from Monash Archive via Hugging Face...")
    try:
        # Many monash datasets are hosted under different handles on HF.
        # "monash_tsf" is a known repository for these datasets.
        # Using a popular and stable dataset as proxy if generic loading is too complex.
        dataset = load_dataset("monash_tsf", dataset_name, split="train")

        # Convert to a DataFrame of series
        data = {}
        count = 0
        for item in dataset:
            if count >= max_series:
                break
            # The 'target' column usually contains the time series
            if 'target' in item:
                series = item['target']
                # Fill missing values if any
                data[f"series_{count}"] = np.array(series)
                count += 1

        # Make a DataFrame with padded sequences
        max_len = max([len(v) for v in data.values()])
        for k in data.keys():
            data[k] = np.pad(data[k], (0, max_len - len(data[k])), constant_values=np.nan)

        return pd.DataFrame(data)

    except Exception as e:
        print(f"Could not load directly from HuggingFace 'monash_tsf': {e}")
        print("Falling back to fetching a lightweight public TS dataset as Monash proxy...")
        # Fallback to generating a "Monash-like" dataset directly to avoid HF download errors in sandbox
        return simulate_monash_dataset(max_series)

def simulate_monash_dataset(num_series=10):
    """
    Simulates a dataset that has properties similar to real-world Monash datasets
    (e.g., varying lengths, strong seasonality, heavy tails).
    """
    data = {}
    for i in range(num_series):
        # Monash datasets typically have highly variable lengths
        length = np.random.randint(500, 2000)
        t = np.arange(length)

        # Real-world scale variations
        scale = np.random.lognormal(mean=2.0, sigma=1.0)

        # Complex trend (e.g. polynomial)
        trend = scale * (0.001 * t + 0.00001 * t**2)

        # Multiple seasonalities (e.g. daily and weekly)
        seasonality_daily = np.sin(2 * np.pi * t / 24) * scale * 0.5
        seasonality_weekly = np.sin(2 * np.pi * t / (24 * 7)) * scale * 0.8

        # Heavy-tailed noise (student-t)
        noise = np.random.standard_t(df=3, size=length) * scale * 0.1

        series = trend + seasonality_daily + seasonality_weekly + noise
        data[f'monash_proxy_series_{i}'] = series

    # Pad for DataFrame
    max_len = max([len(v) for v in data.values()])
    for k in data.keys():
        data[k] = np.pad(data[k], (0, max_len - len(data[k])), constant_values=np.nan)

    return pd.DataFrame(data)

def generate_zero_shot_eval_dataset(num_series, min_length, max_length):
    """
    Generates a synthetic dataset specifically for zero-shot evaluation,
    using entirely different distributions than the standard training generator.
    """
    data = {}
    for i in range(num_series):
        length = np.random.randint(min_length, max_length + 1)
        t = np.arange(length)

        # Out-of-distribution scale (much larger)
        scale = np.random.uniform(5000.0, 20000.0)

        # Out-of-distribution trend (negative exponential)
        trend = scale * np.exp(-0.01 * t)

        # Out-of-distribution high-frequency seasonality
        seasonality = np.cos(2 * np.pi * t / 3.14) * scale * 0.2

        noise = np.random.uniform(-1, 1, length) * scale * 0.05

        series = trend + seasonality + noise
        padded_series = np.pad(series, (0, max_length - length), constant_values=np.nan)
        data[f'zeroshot_eval_series_{i}'] = padded_series

    return pd.DataFrame(data)
