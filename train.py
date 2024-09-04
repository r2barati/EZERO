import torch
from torch.utils.data import DataLoader, TensorDataset
from models.hybrid_model import HybridNBeatsTCNModel
from data_generation import generate_time_series
from data_preprocessing import normalize_series, create_rolling_windows

# Parameters
input_window = 24
forecast_horizon = 4
batch_size = 32
epochs = 50

# Generate synthetic data
num_series = 100
min_length = 50
max_length = 200
time_series_df = generate_time_series(num_series, min_length, max_length)

# Normalize and create rolling windows
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
dataset = TensorDataset(X_all, y_all)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define model, loss, and optimizer
tcn_channels = [64, 128]
tcn_kernel_size = 3
tcn_dropout = 0.2
hidden_units = 128
stack_depth = 3

model = HybridNBeatsTCNModel(input_size=input_window,
                             output_size=forecast_horizon,
                             hidden_units=hidden_units,
                             stack_depth=stack_depth,
                             tcn_channels=tcn_channels,
                             tcn_kernel_size=tcn_kernel_size,
                             tcn_dropout=tcn_dropout)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
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

# Save model
torch.save(model.state_dict(), 'hybrid_model.pth')
