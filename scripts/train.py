import torch
import yaml
import argparse
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.hybrid_model import HybridNBeatsTCNModel  
from data.generator import generate_time_series
from data.monash_loader import load_monash_dataset
from data.preprocessing import normalize_series, create_rolling_windows 

def load_yaml_config(config_file):
    """Load a YAML configuration file."""
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def train_model(model, dataloader, criterion, optimizer, epochs):
    """Training loop for the model."""
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}')

def main(args):
    # Load configurations
    config = load_yaml_config(args.config)
    model_config = config['model']
    training_config = config['training']
    dataset_config = config['dataset']

    # Dataset Loading
    dataset_type = dataset_config.get('type', 'synthetic')
    if dataset_type == 'monash':
        print("Using Monash dataset for training.")
        monash_name = dataset_config.get('monash_name', 'm4_hourly')
        num_series = dataset_config.get('num_series', 10)
        time_series_df = load_monash_dataset(dataset_name=monash_name, max_series=num_series)
    else:
        print("Using synthetic dataset for training.")
        num_series = dataset_config['num_series']
        min_length = dataset_config['min_length']
        max_length = dataset_config['max_length']
        time_series_df = generate_time_series(num_series, min_length, max_length)

    # Normalize and create rolling windows
    input_window = model_config['input_window']
    forecast_horizon = model_config['forecast_horizon']

    means_all, stds_all, X_all, y_all = [], [], [], []
    for col in time_series_df.columns:
        series = time_series_df[col].dropna().values
        X, y, means, stds = create_rolling_windows(series, input_window, forecast_horizon)

        if len(X) > 0:
            X_all.append(X)
            y_all.append(y)
            means_all.append(means)
            stds_all.append(stds)

    X_all = torch.tensor(np.concatenate(X_all), dtype=torch.float32)
    y_all = torch.tensor(np.concatenate(y_all), dtype=torch.float32)
    # Note: means and stds are computed but not strictly required for training the normalized model

    # Create DataLoader
    batch_size = training_config['batch_size']
    dataset = TensorDataset(X_all, y_all)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = HybridNBeatsTCNModel(
        input_size=input_window,
        output_size=forecast_horizon,
        hidden_units=model_config['hidden_units'],
        stack_depth=model_config['stack_depth'],
        tcn_channels=model_config['tcn_channels'],
        tcn_kernel_size=model_config['tcn_kernel_size'],
        tcn_dropout=model_config.get('tcn_dropout', 0.2)
    )

    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_config['learning_rate']))

    # Train the model
    epochs = training_config['epochs']
    train_model(model, dataloader, criterion, optimizer, epochs)

    # Save the model
    torch.save(model.state_dict(), config.get('model_path', 'hybrid_model.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid NBeats + TCN Model Training")
    parser.add_argument('--config', type=str, default='config/config.yaml', help="Path to config file")
    args = parser.parse_args()

    main(args)
