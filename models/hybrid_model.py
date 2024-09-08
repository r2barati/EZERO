import torch
import torch.nn as nn

class NBeatsBlock(nn.Module):
    def __init__(self, input_size, output_size, hidden_units, num_layers=3):
        """
        Single block of the N-BEATS model, consisting of fully connected layers.

        :param input_size: Size of the input time series (number of features).
        :param output_size: Size of the output (forecast horizon).
        :param hidden_units: Number of units in the hidden layers.
        :param num_layers: Number of fully connected layers in each block (default: 3).
        """
        super(NBeatsBlock, self).__init__()
        
        # Define the layers in the block
        layers = [nn.Linear(input_size, hidden_units), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_units, hidden_units), nn.ReLU()]
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(*layers)
        # Final layer to produce the forecast
        self.fc_forecast = nn.Linear(hidden_units, output_size)

        # Optional backcast output (for decomposition into trend/seasonality, not used in simple versions)
        self.fc_backcast = nn.Linear(hidden_units, input_size)  # For residual learning

    def forward(self, x):
        """
        Forward pass through the N-BEATS block. Produces a forecast and optionally a backcast.

        :param x: Input tensor of shape (batch_size, input_size).
        :return: A tuple of (backcast, forecast), where the backcast is the reconstruction of the input
                 and the forecast is the predicted future values.
        """
        # Pass through the fully connected layers
        x = self.fc_layers(x)
        
        # Forecast future values
        forecast = self.fc_forecast(x)
        
        # Backcast is the reconstruction of the input
        backcast = self.fc_backcast(x)
        
        return backcast, forecast

class NBeatsModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_units, stack_depth, num_layers=3):
        """
        N-BEATS model composed of stacked NBeatsBlocks, with residual learning.

        :param input_size: Size of the input time series (number of features).
        :param output_size: Size of the output (forecast horizon).
        :param hidden_units: Number of hidden units in each block.
        :param stack_depth: Number of blocks (depth of the stack).
        :param num_layers: Number of fully connected layers in each block.
        """
        super(NBeatsModel, self).__init__()
        
        # Create a stack of NBeats blocks
        self.blocks = nn.ModuleList(
            [NBeatsBlock(input_size, output_size, hidden_units, num_layers) for _ in range(stack_depth)]
        )

    def forward(self, x):
        """
        Forward pass through the stacked N-BEATS model.

        :param x: Input tensor of shape (batch_size, input_size).
        :return: Final forecast produced by the stacked N-BEATS blocks.
        """
        backcast, forecast = 0, 0
        
        # Each block in the stack refines the forecast, and its residual (backcast) is subtracted
        for block in self.blocks:
            block_backcast, block_forecast = block(x)
            backcast += block_backcast  # Refine the backcast (optional, could be discarded)
            forecast += block_forecast  # Add to the cumulative forecast
            x = x - block_backcast  # Subtract the backcast to create the residual input for the next block
        
        return forecast
