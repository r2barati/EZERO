import torch
import torch.nn as nn
from .tcn import TemporalConvNet  # Assuming TemporalConvNet is in the same directory as tcn.py
from .nbeats import NBeatsBlock   # Assuming NBeatsBlock is in the same directory as nbeats.py

class HybridNBeatsTCNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_units, stack_depth, tcn_channels, tcn_kernel_size, tcn_dropout=0.2):
        """
        Hybrid N-BEATS + TCN Model for Time Series Forecasting.

        Combines a Temporal Convolutional Network (TCN) to extract temporal features from the input time series,
        and stacked N-BEATS blocks to refine the extracted features and produce a final forecast.

        :param input_size: Size of the input time series (number of input features per time step).
        :param output_size: Size of the output forecast (forecast horizon).
        :param hidden_units: Number of hidden units in each N-BEATS block.
        :param stack_depth: Number of stacked N-BEATS blocks.
        :param tcn_channels: List of output channels for each layer in the TCN (defines depth of the TCN).
        :param tcn_kernel_size: Kernel size for each TCN layer.
        :param tcn_dropout: Dropout rate to apply within the TCN layers (default is 0.2).
        """
        super(HybridNBeatsTCNModel, self).__init__()
        
        # Temporal Convolutional Network (TCN) for feature extraction
        self.tcn = TemporalConvNet(1, tcn_channels, kernel_size=tcn_kernel_size, dropout=tcn_dropout)
        
        # Stacked N-BEATS blocks for forecasting
        self.blocks = nn.ModuleList(
            [NBeatsBlock(tcn_channels[-1], output_size, hidden_units) for _ in range(stack_depth)]
        )

        # Initialize weights for the network
        self.init_weights()

    def init_weights(self):
        """
        Initialize the model weights using a normal distribution.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        Forward pass through the Hybrid N-BEATS + TCN Model.

        :param x: Input tensor of shape (batch_size, sequence_length, input_size)
        :return: Final forecast of shape (batch_size, output_size)
        """
        # Ensure the input is in the correct shape for TCN: (batch_size, num_channels, sequence_length)
        x = x.unsqueeze(1)  # Add channel dimension to make it (batch_size, 1, sequence_length)

        # Pass through the Temporal Convolutional Network (TCN)
        x = self.tcn(x)

        # Permute the output to (batch_size, sequence_length, num_channels) for N-BEATS
        x = x.permute(0, 2, 1)

        # Take the last time step's feature map (final temporal representation)
        x = x[:, -1, :]  # Shape: (batch_size, num_channels)

        # Pass through stacked N-BEATS blocks for forecasting
        forecast = 0
        for block in self.blocks:
            _, block_forecast = block(x)  # Only use the forecast, discard the backcast
            forecast += block_forecast  # Accumulate forecasts from each block

        return forecast
