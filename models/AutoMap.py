import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoMap(nn.Module):
    def __init__(self):
        super(AutoMap, self).__init__()
        # Fully connected reconstruction:
        # Input: 5x41 = 205 features, mapped to a 28x28 (784-dimensional) representation.
        self.fc1 = nn.Linear(5 * 41, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 28 * 28)

        # Convolutional refinement:
        # After the FC part, we reshape the 784 vector to (batch, 1, 28, 28)
        # and apply convolutions to refine the reconstruction.
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)   # preserves 28x28
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.deconv = nn.ConvTranspose2d(64, 1, kernel_size=7, stride=1, padding=3)  # preserves 28x28

    def forward(self, x):
        # x is expected to be of shape (batch, 5, 41)
        x = x.view(-1, 5 * 41)            # Flatten to (batch, 205)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                   # Now shape: (batch, 784)
        x = x.view(-1, 1, 28, 28)          # Reshape to (batch, 1, 28, 28)

        # Convolutional refinement:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.deconv(x)                # Final output: (batch, 1, 28, 28)
        return x
