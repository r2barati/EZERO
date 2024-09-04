import torch
import torch.nn as nn

class NBeatsBlock(nn.Module):
    def __init__(self, input_size, output_size, hidden_units):
        super(NBeatsBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, hidden_units)
        self.fc_forecast = nn.Linear(hidden_units, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        forecast = self.fc_forecast(x)
        return forecast

class NBeatsModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_units, stack_depth):
        super(NBeatsModel, self).__init__()
        self.blocks = nn.ModuleList(
            [NBeatsBlock(input_size, output_size, hidden_units) for _ in range(stack_depth)]
        )

    def forward(self, x):
        forecast = 0
        for block in self.blocks:
            forecast += block(x)
        return forecast
