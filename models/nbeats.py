import torch
import torch.nn as nn

class NBeatsBlock(nn.Module):
    def __init__(self, input_size, output_size, hidden_units):
        super(NBeatsBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, hidden_units)
        self.fc4 = nn.Linear(hidden_units, hidden_units)
        
        # Output layers for backcast and forecast
        self.backcast_fc = nn.Linear(hidden_units, input_size)
        self.forecast_fc = nn.Linear(hidden_units, output_size)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        h = self.relu(self.fc4(h))
        
        backcast = self.backcast_fc(h)
        forecast = self.forecast_fc(h)
        
        return backcast, forecast
