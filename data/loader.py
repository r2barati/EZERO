import pandas as pd
import pyarrow.csv as csv
import datasets  
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import yaml
import argparse


class TimeSeriesDataset(Dataset):
    """
    Custom Dataset class to convert time series data into a PyTorch-compatible format.
    """

    def __init__(self, data, target=None):
        """
        Initialize the dataset.
        :param data: Time series data (can be loaded via pandas or Hugging Face datasets).
        :param target: Target values (if applicable, e.g., for supervised learning tasks).
        """
        self.data = data
        self.target = target

    def __len__(self):
        """
        Return the length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.
        :param idx: The index of the sample.
        :return: A tuple (input, target) or just input if no target is provided.
        """
        if self.target is not None:
            return self.data[idx], self.target[idx]
        return self.data[idx]


class DataLoaderHandler:
    def __init__(self, config_file=None):
        """
        Initialize the DataLoaderHandler, optionally using an external config file (YAML).
        """
        self.config = {}
        self.min_patch_size = 2
        self.max_patch_size = 64

        # Predefined patch sizes based on frequency
        self.frequency_patch_map = {
            "seconds": 64,
            "minutes": 32,
            "hours": 16,
            "days": 8,
            "weeks": 4,
            "months": 2
        }

        if config_file:
            self.load_config(config_file)

    def load_config(self, config_file):
        """
        Load configuration settings from a YAML file.
        """
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)

        # Load specific settings from the YAML config
        self.min_patch_size = self.config.get("min_patch_size", 2)
        self.max_patch_size = self.config.get("max_patch_size", 64)
        self.frequency_patch_map.update(self.config.get("frequency_patch_map", {}))

    def load_dataset(self, data_source, dataset_type='csv', split=None):
        """
        Load the dataset based on the provided data_source and dataset_type.
        """
        if dataset_type == 'csv':
            return self.load_csv(data_source)
        elif dataset_type == 'arrow':
            return self.load_arrow(data_source)
        elif dataset_type == 'huggingface':
            return self.load_huggingface(data_source, split)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

    def load_csv(self, file_path):
        """
        Load a dataset from a CSV file.
        """
        return pd.read_csv(file_path)

    def load_arrow(self, file_path):
        """
        Load a dataset from an Apache Arrow file.
        """
        table = csv.read_csv(file_path)
        return table.to_pandas()

    def load_huggingface(self, dataset_name, split=None):
        """
        Load a dataset from Hugging Face Datasets.
        Optionally load a specific split (e.g., 'train', 'test').
        """
        dataset = datasets.load_dataset(dataset_name, split=split)
        return dataset

    def dataset_info(self, dataset):
        """
        Get basic information about the dataset:
        - Length of the dataset
        - Number of variates (columns)
        """
        length = len(dataset)
        if isinstance(dataset, pd.DataFrame):
            num_variates = dataset.shape[1]
        else:
            num_variates = len(dataset.column_names)

        return length, num_variates

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
        
        # Get the patch sizes for predefined frequency
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

    def make_data_patches(self, dataset, frequency):
        """
        Create data patches (chunks) based on the dataset's frequency.
        Patches are created based on the calculated patch size for the given frequency.
        """
        length, num_variates = self.dataset_info(dataset)
        patch_size = self.patch_size_formula(frequency)

        patches = []
        current_size = self.min_patch_size

        while current_size <= patch_size and current_size <= length:
            patches.append(dataset.iloc[:current_size] if isinstance(dataset, pd.DataFrame) else dataset[:current_size])
            current_size *= 2  # Increase patch size exponentially

        return patches

    def get_pytorch_dataloader(self, dataset, batch_size=32, shuffle=True, num_workers=0):
        """
        Create a PyTorch DataLoader from the dataset.
        :param dataset: Time series dataset (can be loaded via Hugging Face or CSV).
        :param batch_size: Size of each batch to load.
        :param shuffle: Whether to shuffle the dataset after each epoch.
        :param num_workers: Number of subprocesses to use for data loading.
        :return: PyTorch DataLoader instance.
        """
        ts_dataset = TimeSeriesDataset(dataset)
        return DataLoader(ts_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def main():
    """
    Main function to run the DataLoader from the terminal using a YAML configuration file.
    """
    parser = argparse.ArgumentParser(description="DataLoader for multiple dataset types and patch generation.")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')

    args = parser.parse_args()

    # Initialize DataLoaderHandler with the configuration file
    dataloader_handler = DataLoaderHandler(args.config)

    # Load the dataset based on the configuration file
    dataset_type = dataloader_handler.config.get('dataset_type', 'csv')
    data_source = dataloader_handler.config.get('file_path', None)
    if data_source:
        dataset = dataloader_handler.load_dataset(data_source, dataset_type=dataset_type)
    else:
        raise ValueError("Data source must be specified in the configuration file.")

    # Get dataset frequency from the configuration file
    frequency = dataloader_handler.config.get('frequency', None)
    if not frequency:
        raise ValueError("Dataset frequency must be specified in the configuration file.")

    # Create data patches (chunks) based on the dataset and frequency
    patches = dataloader_handler.make_data_patches(dataset, frequency)

    # Get a PyTorch DataLoader with the generated patches
    pytorch_dataloader = dataloader_handler.get_pytorch_dataloader(patches)

    print(f"PyTorch DataLoader is ready with {len(pytorch_dataloader)} batches.")

if __name__ == "__main__":
    main()
