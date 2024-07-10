from .base import BaseGNNModel

from .gcn import GCN
from .gcn2 import GCN2
from .sage import SAGE
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPWithLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.0, batch_norm=False, residual=False):
        super(MLPWithLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_channels, batch_first=True)
        self.linear2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual

        if batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_channels)
            self.bn2 = nn.BatchNorm1d(out_channels)

        self.hidden_state = None

    def forward(self, x):
        # Add a batch dimension if not present
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Shape (1, seq_len, in_channels)

        # Initialize hidden state if not set
        if self.hidden_state is None:
            h_0 = torch.zeros(1, x.size(0), self.lstm.hidden_size).to(x.device)  # (num_layers, batch, hidden_size)
            c_0 = torch.zeros(1, x.size(0), self.lstm.hidden_size).to(x.device)
        else:
            h_0, c_0 = self.hidden_state

        lstm_out, self.hidden_state = self.lstm(x, (h_0, c_0))
        x = lstm_out[:, -1, :]  # Use the output of the last time step
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

