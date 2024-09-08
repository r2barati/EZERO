import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

# Helper class to handle Chomping of extra padding (useful for causal convolutions)
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        """
        Chomps off the extra padding introduced by the convolution operation to maintain 
        the causal nature of the convolution (i.e., no information leakage from future).
        
        :param chomp_size: Size to chomp off from the output tensor.
        """
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Chomps the tensor to ensure that no future information is used in causal convolutions.
        """
        return x[:, :, :-self.chomp_size].contiguous()

# A single block of temporal convolution with residual connections.
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        A temporal convolution block consisting of two convolutional layers with residual connection.
        Includes weight normalization, dropout, and ReLU activations.
        
        :param n_inputs: Number of input channels.
        :param n_outputs: Number of output channels.
        :param kernel_size: Size of the convolutional kernel.
        :param stride: Stride for the convolution.
        :param dilation: Dilation factor for the convolution.
        :param padding: Padding to ensure the output length matches the input length.
        :param dropout: Dropout rate for regularization.
        """
        super(TemporalBlock, self).__init__()
        # First convolutional layer followed by Chomp and ReLU activation
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second convolutional layer
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Sequential block combining convolution, chomp, ReLU, and dropout
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # Residual connection to match input/output dimensions (downsample if necessary)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

        # Initialize the weights of the convolutional layers
        self.init_weights()

    def init_weights(self):
        """
        Initializes the weights of the convolutional layers with a normal distribution.
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        Forward pass through the TemporalBlock.
        Applies two convolutions with ReLU, dropout, and a residual connection.
        
        :param x: Input tensor of shape (batch_size, channels, sequence_length)
        :return: Output tensor after applying the temporal block.
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)  # Residual connection followed by ReLU activation

# Temporal Convolutional Network (TCN) architecture composed of multiple TemporalBlocks
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        Temporal Convolutional Network (TCN) model for time series forecasting.
        
        :param num_inputs: Number of input channels (e.g., features in the time series).
        :param num_channels: List of output channels for each layer (defines the depth of the network).
        :param kernel_size: Size of the convolutional kernel (default is 2).
        :param dropout: Dropout rate for regularization.
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        # Build a stack of TemporalBlocks
        for i in range(num_levels):
            dilation_size = 2 ** i  # Exponentially increasing dilation
            in_channels = num_inputs if i == 0 else num_channels[i-1]  # First layer uses input channels
            out_channels = num_channels[i]  # Output channels for each block
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size, padding=(kernel_size-1) * dilation_size,
                                     dropout=dropout)]
        
        # Combine all the temporal blocks into a sequential network
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the Temporal Convolutional Network (TCN).
        
        :param x: Input tensor of shape (batch_size, num_channels, sequence_length)
        :return: Output tensor after applying all the temporal blocks.
        """
        return self.network(x)
