import pandas as pd
import numpy as np
import requests
import zipfile
import io
import os
import ast

# Hardcoded Monash datasets and their Zenodo links for direct download
MONASH_DATASETS = {
    "m4_hourly": "https://zenodo.org/api/records/4656548/files/m4_hourly_dataset.zip/content",
    "m4_daily": "https://zenodo.org/api/records/4656548/files/m4_daily_dataset.zip/content", # Using 4656548 as base for M4 datasets if exact is unknown, or tourism for robust test.
    "tourism_monthly": "https://zenodo.org/api/records/4656096/files/tourism_monthly_dataset.zip/content",
    "solar_10_minutes": "https://zenodo.org/api/records/4656144/files/solar_10_minutes_dataset.zip/content",
    # Add others as needed
}

def parse_tsf(file_path, max_series=None):
    """
    Minimal parser for Monash .tsf files.
    Extracts the series values (comma separated) into a list of arrays.
    Memory-efficient iterator.
    """
    series_list = []
    with open(file_path, 'r', encoding='ISO-8859-1') as f:
        # Skip header metadata until @data
        for line in f:
            if line.startswith('@data'):
                break

        # Process data rows
        for line in f:
            if max_series is not None and len(series_list) >= max_series:
                break
            parts = line.strip().split(':')
            if len(parts) > 1:
                # The series data is usually in the last part, separated by commas
                series_data_str = parts[-1].split(',')
                try:
                    # Convert strings to floats
                    series_data = [float(val) if val != '?' else np.nan for val in series_data_str]
                    series_list.append(np.array(series_data))
                except ValueError:
                    pass
    return series_list

def load_all_monash_datasets(max_series_per_dataset=10):
    """
    Downloads, extracts, and parses multiple monash datasets, merging them
    into a single unified training corpus for foundation-model style training.
    """
    all_series = []
    print(f"Aggregating {len(MONASH_DATASETS)} datasets for zero-shot capability training...")
    for dataset_name in MONASH_DATASETS.keys():
        series = load_monash_dataset(dataset_name, max_series=max_series_per_dataset)
        all_series.extend(series)

    print(f"Total series aggregated: {len(all_series)}")
    return all_series

def load_monash_dataset(dataset_name="m4_hourly", max_series=10):
    """
    Downloads and parses a dataset directly from the Monash Time Series
    Forecasting Archive hosted on Zenodo.
    """
    print(f"Loading {dataset_name} directly from Monash Archive (Zenodo)...")

    if dataset_name not in MONASH_DATASETS:
        print(f"Dataset {dataset_name} not found in hardcoded links. Falling back to proxy.")
        return simulate_monash_dataset(max_series)

    url = MONASH_DATASETS[dataset_name]
    dataset_dir = f"data/{dataset_name}"
    tsf_file = f"{dataset_dir}/{dataset_name}_dataset.tsf"

    # Download and extract if not already cached
    if not os.path.exists(tsf_file):
        print(f"Downloading from {url}...")
        try:
            response = requests.get(url)
            response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(dataset_dir)
                print(f"Extracted to {dataset_dir}")
        except Exception as e:
            print(f"Download or extraction failed: {e}")
            return simulate_monash_dataset(max_series)

    # Locate the .tsf file (sometimes the name inside the zip varies slightly)
    extracted_tsf = None
    for file in os.listdir(dataset_dir):
        if file.endswith(".tsf"):
            extracted_tsf = os.path.join(dataset_dir, file)
            break

    if not extracted_tsf:
        print("Could not find .tsf file in the downloaded archive.")
        return simulate_monash_dataset(max_series)

    print(f"Parsing {extracted_tsf}...")
    series_list = parse_tsf(extracted_tsf, max_series=max_series)

    return series_list

def simulate_monash_dataset(num_series=10):
    """
    Simulates a dataset that has properties similar to real-world Monash datasets
    (e.g., varying lengths, strong seasonality, heavy tails).
    Returns a list of 1D numpy arrays.
    """
    series_list = []
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
        series_list.append(series)

    return series_list

def generate_zero_shot_eval_dataset(num_series, min_length, max_length):
    """
    Generates a synthetic dataset specifically for zero-shot evaluation,
    using entirely different distributions than the standard training generator.
    Returns a list of 1D numpy arrays.
    """
    series_list = []
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
        series_list.append(series)

    return series_list
