import torch
import torch.nn as nn
from .tcn import TemporalConvNet
from .nbeats import NBeatsBlock

class HybridNBeatsTCNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_units, stack_depth, tcn_channels, tcn_kernel_size, tcn_dropout):
        super(HybridNBeatsTCNModel, self).__init__()
        self.tcn = TemporalConvNet(1, tcn_channels, kernel_size=tcn_kernel_size, dropout=tcn_dropout)
        self.blocks = nn.ModuleList(
            [NBeatsBlock(tcn_channels[-1], output_size, hidden_units) for _ in range(stack_depth)]
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)
        x = x[:, -1, :]

        forecast = 0
        for block in self.blocks:
            forecast += block(x)
        return forecast
