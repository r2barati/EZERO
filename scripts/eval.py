import torch
import argparse
import yaml
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models.hybrid_model import HybridNBeatsTCNModel 
from data.generator import generate_time_series
from data.monash_loader import generate_zero_shot_eval_dataset
from data.preprocessing import normalize_series, create_rolling_windows  

def load_yaml_config(config_file):
    """Load a YAML configuration file."""
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def load_model(model_class, model_path, *args, **kwargs):
    """Load a saved model from a checkpoint."""
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def evaluate_model(model, dataloader):
    """Evaluate the model on the test set."""
    model.eval()  # Set the model to evaluation mode
    all_preds, all_targets = [], []
    
    with torch.no_grad():  # Disable gradient computation
        for inputs, targets in dataloader:
            outputs = model(inputs)
            all_preds.append(outputs.numpy())
            all_targets.append(targets.numpy())
    
    # Convert lists to numpy arrays
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate evaluation metrics
    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)

    return mse, mae

def main(args):
    # Load configurations
    config = load_yaml_config(args.config)
    model_config = config['model']
    evaluation_config = config['evaluation']
    dataset_config = config['dataset']

    # Explicit Zero-Shot evaluation Dataset setup
    # Using entirely different distributions and scales
    eval_num_series = evaluation_config.get('zero_shot_num_series', 20)
    eval_min_length = evaluation_config.get('zero_shot_min_length', 100)
    eval_max_length = evaluation_config.get('zero_shot_max_length', 300)

    print("Generating explicit Zero-Shot Evaluation Dataset...")
    series_list = generate_zero_shot_eval_dataset(eval_num_series, eval_min_length, eval_max_length)

    # Normalize and create rolling windows
    input_window = model_config['input_window']
    forecast_horizon = model_config['forecast_horizon']

    means_all, stds_all, X_all, y_all = [], [], [], []
    for series in series_list:
        series = series[~np.isnan(series)]
        if len(series) < input_window + forecast_horizon:
            continue

        X, y, means, stds = create_rolling_windows(series, input_window, forecast_horizon)

        if len(X) > 0:
            X_all.append(X)
            y_all.append(y)
            means_all.append(means)
            stds_all.append(stds)

    X_all = torch.tensor(np.concatenate(X_all), dtype=torch.float32)
    y_all = torch.tensor(np.concatenate(y_all), dtype=torch.float32)
    means_tensor = torch.tensor(np.concatenate(means_all), dtype=torch.float32)
    stds_tensor = torch.tensor(np.concatenate(stds_all), dtype=torch.float32)

    # Create DataLoader for evaluation
    batch_size = evaluation_config['batch_size']
    # Include means and stds in the dataset for denormalization
    dataset = TensorDataset(X_all, y_all, means_tensor, stds_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model weights from the path specified in evaluation_config.yaml
    model_path = config.get('model_path', 'hybrid_model.pth')
    model = load_model(
        HybridNBeatsTCNModel,
        model_path,
        input_size=input_window,
        output_size=forecast_horizon,
        hidden_units=model_config['hidden_units'],
        stack_depth=model_config['stack_depth'],
        tcn_channels=model_config['tcn_channels'],
        tcn_kernel_size=model_config['tcn_kernel_size'],
        tcn_dropout=model_config.get('tcn_dropout', 0.2)
    )

    # Evaluate the model on the test data
    # Redefine evaluate_model to handle denormalization
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets, window_means, window_stds in dataloader:
            outputs = model(inputs)

            # Denormalize predictions and targets using instance statistics
            outputs_denorm = outputs * window_stds.unsqueeze(-1) + window_means.unsqueeze(-1)
            targets_denorm = targets * window_stds.unsqueeze(-1) + window_means.unsqueeze(-1)

            all_preds.append(outputs_denorm.numpy())
            all_targets.append(targets_denorm.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate evaluation metrics in original scale
    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)

    print(f"Evaluation Results on Denormalized Data (Original Scale):\nMSE: {mse:.4f}\nMAE: {mae:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid NBeats + TCN Model Evaluation")
    parser.add_argument('--config', type=str, default='config/config.yaml', help="Path to config file")
    args = parser.parse_args()

    main(args)
