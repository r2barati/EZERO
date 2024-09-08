import torch
import argparse
import yaml
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models.hybrid_model import HybridNBeatsTCNModel 
from data.dataloader import generate_time_series 
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
    model_config = load_yaml_config(args.model_config)
    evaluation_config = load_yaml_config(args.evaluation_config)

    # Generate test data (or load it, depending on your application)
    num_series = evaluation_config['dataset']['num_series']
    min_length = evaluation_config['dataset']['min_length']
    max_length = evaluation_config['dataset']['max_length']
    time_series_df = generate_time_series(num_series, min_length, max_length)

    # Normalize and create rolling windows
    input_window = model_config['input_window']
    forecast_horizon = model_config['forecast_horizon']

    means, stds, X_all, y_all = [], [], [], []
    for col in time_series_df.columns:
        series = time_series_df[col].dropna().values
        normalized_series, mean, std = normalize_series(series)
        X, y = create_rolling_windows(normalized_series, input_window, forecast_horizon)

        if len(X) > 0:
            X_all.append(X)
            y_all.append(y)
        means.append(mean)
        stds.append(std)

    X_all = torch.tensor(np.concatenate(X_all), dtype=torch.float32)
    y_all = torch.tensor(np.concatenate(y_all), dtype=torch.float32)

    # Create DataLoader for evaluation
    batch_size = evaluation_config['evaluation']['batch_size']
    dataset = TensorDataset(X_all, y_all)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model weights from the path specified in evaluation_config.yaml
    model_path = evaluation_config['model_path']
    model = load_model(
        HybridNBeatsTCNModel,
        model_path,
        input_size=input_window,
        output_size=forecast_horizon,
        hidden_units=model_config['hidden_units'],
        stack_depth=model_config['stack_depth'],
        tcn_channels=model_config['tcn_channels'],
        tcn_kernel_size=model_config['tcn_kernel_size'],
        tcn_dropout=model_config['tcn_dropout']
    )

    # Evaluate the model on the test data
    mse, mae = evaluate_model(model, dataloader)

    print(f"Evaluation Results:\nMSE: {mse:.4f}\nMAE: {mae:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid NBeats + TCN Model Evaluation")
    parser.add_argument('--model_config', type=str, default='../configs/model_config.yaml', help="Path to model config file")
    parser.add_argument('--evaluation_config', type=str, default='../configs/evaluation_config.yaml', help="Path to evaluation config file")
    args = parser.parse_args()

    main(args)
