import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from collections import OrderedDict
import numpy as np

class LSTMPolicy(nn.Module):
    """
    Baseline LSTM Architecture
    """
    def __init__(self, input_size, output_size, lstm_size=256):
        super(LSTMPolicy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.lstm_size = lstm_size

        lstm_input_size = self.input_size + self.output_size + 3
        self.lstm = nn.LSTMCell(lstm_input_size, hidden_size=self.lstm_size)
        self.pi = nn.Linear(self.lstm_size, self.output_size)
        self.v = nn.Linear(self.lstm_size, 1)

    def forward(self, x, hx, cx, embed):
        output = torch.cat((x, embed), dim=1)
        h, c = self.lstm(output, (hx, cx))
        return Categorical(logits=self.pi(h)), self.v(h), h, c

class GRUPolicy(nn.Module):
    """
    Baseline GRU Architecture
    """
    def __init__(self, input_size, output_size, lstm_size=256, D=1):
        super(GRUPolicy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.lstm_size = lstm_size
        self.D = D

        lstm_input_size = self.input_size + self.output_size + 3
        self.cell_list = [nn.GRUCell(lstm_input_size, hidden_size=self.lstm_size)]
        for d in range(1, self.D):
            self.cell_list.append(nn.GRUCell(self.lstm_size, self.lstm_size))
        self.cell_list = nn.ModuleList(self.cell_list)
        self.pi = nn.Linear(self.lstm_size, self.output_size)
        self.v = nn.Linear(self.lstm_size, 1)

    def forward(self, x, hx, embed):
        output = torch.cat((x, embed), dim=1)
        h_out = []
        for d in range(self.D):
            output = self.cell_list[d](output, hx[d])
            h_out.append(output)
        return Categorical(logits=self.pi(output)), self.v(output), torch.stack(h_out)

