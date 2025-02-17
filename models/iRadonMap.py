import torch
import torch.nn as nn
import torch.nn.functional as F

class FilteringLayer(nn.Module):
    """ Learnable filtering layer to simulate the FBP filtering process """
    def __init__(self, input_size, hidden_size=256):
        super(FilteringLayer, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size, bias=False)  # Fixed dimensions

    def forward(self, x):
        return self.fc(x)

class SinusoidalBackProjectionLayer(nn.Module):
    """ Learnable sinusoidal back-projection layer """
    def __init__(self, input_size, output_size):
        super(SinusoidalBackProjectionLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size, bias=False)  # Ensure matching dims

    def forward(self, x):
        return self.fc(x)

class ResidualBlock(nn.Module):
    """ ResNet-based residual block """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + residual)

class ResNetCNN(nn.Module):
    """ ResNet structure for CT image optimization """
    def __init__(self, in_channels, num_blocks=5):
        super(ResNetCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_blocks)])
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.res_blocks(out)
        out = self.conv2(out)
        return out

class iRadonMap(nn.Module):
    def __init__(self, input_size=205, output_size=784, num_blocks=5, hidden_size=256):
        super(iRadonMap, self).__init__()
        self.filtering = FilteringLayer(input_size, hidden_size)
        self.back_projection = SinusoidalBackProjectionLayer(hidden_size, output_size)
        self.cnn = ResNetCNN(1, num_blocks=num_blocks)

    def forward(self, x):
        x = x.squeeze(1)  # Remove the extra dimension -> [batch_size, 5, 41]
        x = x.view(x.shape[0], -1)  # Flatten to [batch_size, 205]
        x = self.filtering(x)  # Filtering step
        x = self.back_projection(x)  # Back-projection step
        x = x.view(x.shape[0], 1, 28, 28)  # Reshape to image format (batch_size, 1, 28, 28)
        x = self.cnn(x)  # CNN enhancement
        return x

