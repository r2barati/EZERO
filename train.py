import torch
import yaml
import argparse
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.hybrid_model import HybridNBeatsTCNModel  
from data.dataloader import generate_time_series  
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
    model_config = load_yaml_config(args.model_config)
    training_config = load_yaml_config(args.training_config)

    # Data generation
    num_series = training_config['num_series']
    min_length = training_config['min_length']
    max_length = training_config['max_length']
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
        tcn_dropout=model_config['tcn_dropout']
    )

    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config['learning_rate'])

    # Train the model
    epochs = training_config['epochs']
    train_model(model, dataloader, criterion, optimizer, epochs)

    # Save the model
    torch.save(model.state_dict(), 'hybrid_model.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid NBeats + TCN Model Training")
    parser.add_argument('--model_config', type=str, default='../configs/model_config.yaml', help="Path to model config file")
    parser.add_argument('--training_config', type=str, default='../configs/training_config.yaml', help="Path to training config file")
    args = parser.parse_args()

    main(args)
