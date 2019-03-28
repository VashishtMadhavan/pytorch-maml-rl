import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from collections import OrderedDict
from maml_rl.policies.policy import ConvGRUCell
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

class GRUVaePolicy(nn.Module):
    """
    Baseline GRU Architecture
    """
    def __init__(self, output_size, encoder_model, lstm_size=256, D=1):
        super(GRUVaePolicy, self).__init__()
        self.output_size = output_size
        self.lstm_size = lstm_size
        self.D = D
        self.encoder = encoder_model

        lstm_input_size = 32 + self.output_size + 2
        self.cell_list = [nn.GRUCell(lstm_input_size, hidden_size=self.lstm_size)]
        for d in range(1, self.D):
            self.cell_list.append(nn.GRUCell(self.lstm_size, self.lstm_size))
        self.cell_list = nn.ModuleList(self.cell_list)
        self.pi = nn.Linear(self.lstm_size, self.output_size)
        self.v = nn.Linear(self.lstm_size, 1)

    def forward(self, x, hx, embed):
        output = x.permute(0, 3, 1, 2)
        mu, sigma = self.encoder.encode(output)
        output = self.encoder.sample_latent(mu, sigma)
        output = torch.cat((output, embed), dim=1)
        h_out = []
        for d in range(self.D):
            output = self.cell_list[d](output, hx[d])
            h_out.append(output)
        return Categorical(logits=self.pi(output)), self.v(output), torch.stack(h_out)

class GRUPolicy(nn.Module):
    """
    Baseline GRU Architecture
    """
    def __init__(self, input_size, output_size, lstm_size=256, D=1, N=1):
        super(GRUPolicy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.lstm_size = lstm_size
        self.D = D
        self.N = N

        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=1)
        lstm_input_shape = (2,2)

        self.cell_list = [ConvGRUCell(input_size=lstm_input_shape,
            input_dim=16, hidden_dim=self.lstm_size, kernel_size=3)]
        for d in range(1, self.D):
            self.cell_list.append(ConvGRUCell(input_size=lstm_input_shape,
                input_dim=self.lstm_size, hidden_dim=self.lstm_size, kernel_size=3))
        self.cell_list = nn.ModuleList(self.cell_list)
        self.pi = nn.Linear(2 * 2 * lstm_size, self.output_size)
        self.v = nn.Linear(2 * 2 * lstm_size, 1)

    def forward(self, x, hx, embed):
        output = x.unsqueeze(1)
        output = F.relu(self.conv1(output))
        #output = output.view(output.size(0), -1)
        #output = torch.cat((output, embed), dim=1)
        h_in = hx
        for n in range(self.N):
            inner_out = output; h_out = []
            for d in range(self.D):
                inner_out = self.cell_list[d](inner_out, h_in[d])
                h_out.append(inner_out)
            h_in = h_out
        output = inner_out.view(inner_out.size(0), -1)
        return Categorical(logits=self.pi(output)), self.v(output), torch.stack(h_out)

class FFPolicy(nn.Module):
    """
    Baseline GRU Architecture
    """
    def __init__(self, input_size, output_size, D=1):
        super(FFPolicy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.D = D

        self.fc = nn.Linear(self.input_size, 64)
        self.fc2 = nn.Linear(64, 64)

        self.pi = nn.Linear(64, self.output_size)
        self.v = nn.Linear(64, 1)

    def forward(self, x):
        output = F.relu(self.fc(x))
        output = F.relu(self.fc2(output))
        return Categorical(logits=self.pi(output)), self.v(output)

