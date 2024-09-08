import numpy as np
import pandas as pd
import yaml

class TimeSeriesProcessor:
    def __init__(self, config_file=None):
        """
        Initialize the processor, optionally using an external config file (YAML).
        """
        self.config = {}

        # Default settings if not provided in the YAML
        self.input_window = 512  # Default max context size (window)
        self.forecast_horizon = 720  # Default forecast horizon (future time steps)
        self.impute_method = "mean"  # Method for missing value imputation
        self.min_patch_size = 2
        self.max_patch_size = 64
        self.frequency_patch_map = {
            "seconds": 64,
            "minutes": 32,
            "hours": 16,
            "days": 8,
            "weeks": 4,
            "months": 2
        }

        # Load config from YAML if provided
        if config_file:
            self.load_config(config_file)

    def load_config(self, config_file):
        """
        Load configuration settings from a YAML file.
        """
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)

        # Load specific configurations if available in YAML
        self.input_window = self.config.get("input_window", 512)
        self.forecast_horizon = self.config.get("forecast_horizon", 720)
        self.impute_method = self.config.get("impute_method", "mean")
        self.min_patch_size = self.config.get("min_patch_size", 2)
        self.max_patch_size = self.config.get("max_patch_size", 64)
        self.frequency_patch_map.update(self.config.get("frequency_patch_map", {}))

    def handle_missing_values(self, series):
        """
        Handle missing values in the series based on the chosen imputation method.
        Options: 'mean', 'median', 'ffill', 'bfill'.
        """
        if self.impute_method == "mean":
            return series.fillna(series.mean())
        elif self.impute_method == "median":
            return series.fillna(series.median())
        elif self.impute_method == "ffill":
            return series.fillna(method='ffill')
        elif self.impute_method == "bfill":
            return series.fillna(method='bfill')
        else:
            raise ValueError(f"Unknown impute method: {self.impute_method}")

    def normalize_series(self, series):
        """
        Normalize the series (z-score normalization) and return the normalized series, mean, and standard deviation.
        """
        mean = series.mean()
        std = series.std()
        return (series - mean) / std, mean, std

    def create_rolling_windows(self, series):
        """
        Create rolling windows based on input window and forecast horizon.
        Ensure that the input window is capped by self.input_window and forecast horizon by self.forecast_horizon.
        """
        if len(series) < self.input_window + self.forecast_horizon:
            return np.array([]), np.array([])

        X, y = [], []
        for i in range(len(series) - self.input_window - self.forecast_horizon + 1):
            X.append(series[i:i + self.input_window])
            y.append(series[i + self.input_window:i + self.input_window + self.forecast_horizon])
        return np.array(X), np.array(y)

    def interpolate_patch_size(self, freq_value, freq_lower, freq_upper, patch_lower, patch_upper):
        """
        Linear interpolation to calculate patch size for intermediate values.
        """
        ratio = (freq_value - freq_lower) / (freq_upper - freq_lower)
        interpolated_patch_size = patch_lower + (patch_upper - patch_lower) * ratio
        return round(interpolated_patch_size)

    def patch_size_formula(self, frequency):
        """
        Determine patch size based on predefined frequencies or interpolate for intermediate values.
        """
        # Predefined frequencies and corresponding patch sizes
        freq_values = {
            "seconds": 1,
            "minutes": 60,
            "hours": 3600,
            "days": 86400,
            "weeks": 604800,
            "months": 2592000  # Approx 30 days in seconds
        }
        
        patch_sizes = self.frequency_patch_map

        if frequency in patch_sizes:
            return patch_sizes[frequency]
        
        # Interpolate patch size for in-between values
        freq_value = freq_values.get(frequency)
        if freq_value is None:
            raise ValueError(f"Unknown frequency: {frequency}")

        # Find the two nearest predefined frequency values for interpolation
        freqs = sorted(freq_values.items(), key=lambda x: x[1])
        for i in range(len(freqs) - 1):
            freq_lower, val_lower = freqs[i]
            freq_upper, val_upper = freqs[i + 1]
            if val_lower <= freq_value <= val_upper:
                return self.interpolate_patch_size(
                    freq_value, val_lower, val_upper, patch_sizes[freq_lower], patch_sizes[freq_upper]
                )

        return self.min_patch_size  # Default to min patch size if no match

    def process_series(self, series, frequency):
        """
        High-level method to process the series: handle missing values, normalize, and create rolling windows.
        Handles patching based on frequency and ensures the context window doesn't exceed input_window.
        """
        # Handle missing values
        clean_series = self.handle_missing_values(series)

        # Normalize the series
        normalized_series, mean, std = self.normalize_series(clean_series)

        # Determine patch size based on frequency
        patch_size = self.patch_size_formula(frequency)

        # Create rolling windows within the max context window and forecast horizon
        X, y = self.create_rolling_windows(normalized_series)

        return X, y, mean, std

    def apply_context_limits(self, dataset, frequency):
        """
        Apply context window size limits (max 512 input window) and adjust for patch size.
        :param dataset: Input dataset, can be pandas DataFrame or Hugging Face dataset.
        :param frequency: Frequency of the dataset.
        """
        # Get patch size based on frequency
        patch_size = self.patch_size_formula(frequency)

        if patch_size > self.input_window:
            patch_size = self.input_window  # Cap the patch size at input_window (max 512)

        patches = []
        current_size = self.min_patch_size

        while current_size <= patch_size:
            patches.append(dataset.iloc[:current_size] if isinstance(dataset, pd.DataFrame) else dataset[:current_size])
            current_size *= 2

        return patches

