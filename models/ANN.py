import torch
import torch.nn as nn
import torch.nn.functional as F

class FullRecoNet(nn.Module):
    def __init__(self, input_size, hidden_dim=256, output_dim=28*28):
        """
        Args:
            input_size (int): Dimension of the flattened input (e.g. 5*41, 72*41, etc.).
            hidden_dim (int): Dimension of the hidden layers.
            output_dim (int): Dimension of the output (default is 28*28).
        """
        super(FullRecoNet, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.lin1 = nn.Linear(input_size, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Flatten the input to [batch_size, input_size]
        x = x.view(-1, self.input_size)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)

        # Reshape to 1 x 28 x 28 if your output is always meant to be an image of shape 28x28
        # For a more generic shape, you'd also parameterize the final dimensions
        return x.view(-1, 1, 28, 28)
