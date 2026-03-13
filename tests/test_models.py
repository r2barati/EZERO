import torch
import pytest
from models.nbeats import NBeatsBlock
from models.tcn import Chomp1d, TemporalBlock, TemporalConvNet
from models.hybrid_model import HybridNBeatsTCNModel

def test_chomp1d():
    # Input shape: (batch_size, channels, sequence_length)
    x = torch.arange(24).view(2, 3, 4).float()
    chomp_size = 2
    chomper = Chomp1d(chomp_size)
    output = chomper(x)

    # Sequence length should drop by chomp_size
    assert output.shape == (2, 3, 4 - chomp_size)
    assert torch.allclose(output, x[:, :, :-chomp_size])

def test_temporal_block():
    batch_size = 4
    n_inputs = 3
    n_outputs = 5
    seq_len = 10

    x = torch.randn(batch_size, n_inputs, seq_len)

    block = TemporalBlock(n_inputs, n_outputs, kernel_size=2, stride=1, dilation=2, padding=2, dropout=0.0)
    output = block(x)

    # Causal TemporalBlock with Chomp should preserve sequence length exactly
    assert output.shape == (batch_size, n_outputs, seq_len)

def test_temporal_conv_net():
    batch_size = 4
    num_inputs = 1
    seq_len = 16
    num_channels = [16, 32, 64]

    x = torch.randn(batch_size, num_inputs, seq_len)

    tcn = TemporalConvNet(num_inputs, num_channels, kernel_size=3)
    output = tcn(x)

    # Should output the last defined channel depth while preserving length
    assert output.shape == (batch_size, num_channels[-1], seq_len)

def test_nbeats_block():
    batch_size = 8
    input_size = 64 # Let's say it's taking the final TCN output embedding
    output_size = 12 # Forecast horizon
    hidden_units = 32

    x = torch.randn(batch_size, input_size)

    block = NBeatsBlock(input_size, output_size, hidden_units)
    backcast, forecast = block(x)

    assert backcast.shape == (batch_size, input_size)
    assert forecast.shape == (batch_size, output_size)

def test_hybrid_nbeats_tcn_model():
    batch_size = 16
    input_size = 48  # Lookback window
    output_size = 24 # Forecast horizon
    hidden_units = 128
    stack_depth = 3
    tcn_channels = [32, 64, 128]
    tcn_kernel_size = 2

    x = torch.randn(batch_size, input_size)

    model = HybridNBeatsTCNModel(
        input_size=input_size,
        output_size=output_size,
        hidden_units=hidden_units,
        stack_depth=stack_depth,
        tcn_channels=tcn_channels,
        tcn_kernel_size=tcn_kernel_size
    )

    forecast = model(x)

    # Output should simply be the forecast values
    assert forecast.shape == (batch_size, output_size)
