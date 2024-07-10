from .base import BaseGNNModel

from .gcn import GCN
from .gcn2 import GCN2
from .sage import SAGE
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPWithContext(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.0, batch_norm=False, residual=False):
        super(MLPWithContext, self).__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, out_channels)
        self.hidden_state = None
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual

        if batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_channels)
            self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # Ensure hidden_state has the same shape as x
        if self.hidden_state is None or self.hidden_state.size() != x.size():
            self.hidden_state = torch.zeros_like(x)

        x = self.linear1(x + self.hidden_state)
        self.hidden_state = x
        if self.batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear2(x)
        if self.batch_norm:
            x = self.bn2(x)
        return x

    def reset_hidden_state(self):
        self.hidden_state = None