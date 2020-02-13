import torch
import torch.nn as nn
import torch.nn.functional as F


class MCLR(nn.Module):
    def __init__(self):
        super(MCLR, self).__init__()
        self.fc1 = nn.Linear(40, 1)

    def forward(self, x):
        # Assume that x is of shape (40, )
        x = self.fc1(x)
        # Squeeze the last dimension of x, which is 1 (to make it
        # compatible with y)
        output = x.squeeze(1)
        return output
