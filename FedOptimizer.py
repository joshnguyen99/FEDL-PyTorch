import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer


class MySGD(Optimizer):
    def __init__(self, params, lr=0.01):
        defaults = dict(lr=lr)
        super(MySGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(-group['lr'], d_p)

        return loss


class FEDLOptimizer(Optimizer):
    def __init__(self, lr=0.01, ):
        pass