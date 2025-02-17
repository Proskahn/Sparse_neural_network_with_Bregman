import torch
import torch.nn as nn
import torch.nn.functional as F

class FullRecoNet(nn.Module):
    def __init__(self):
        super(FullRecoNet, self).__init__()
        self.lin1 = nn.Linear(5 * 41, 256)
        self.lin2 = nn.Linear(256, 256)
        self.lin3 = nn.Linear(256, 28 ** 2)

    def forward(self, x):
        x = F.relu(self.lin1(x.view(-1, 5 * 41)))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x.view(-1, 1, 28, 28)