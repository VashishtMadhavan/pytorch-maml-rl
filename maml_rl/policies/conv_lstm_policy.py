import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from collections import OrderedDict
from maml_rl.policies.policy import ImpalaCnn, NatureCnn
import numpy as np

class ConvLSTMPolicy(nn.Module):
    """
    Baseline LSTM Architecture
    """
    def __init__(self, input_size, output_size, use_bn=False,
                cnn_type='nature', D=1, N=1):
        super(ConvLSTMPolicy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.use_bn = use_bn
        self.cnn_type = cnn_type
        self.D = D
        self.N = N

        if self.cnn_type == 'nature':
            self.encoder = NatureCnn(input_size=input_size, use_bn=use_bn)
            hidden_size = 512
        elif self.cnn_type == 'impala':
            self.encoder = ImpalaCnn(input_size=input_size, use_bn=use_bn)
            hidden_size = 256

        lstm_input_size = hidden_size + self.output_size + 2
        self.cell_list = [nn.LSTMCell(lstm_input_size, hidden_size=256)]
        for d in range(1, self.D):
            self.cell_list.append(nn.LSTMCell(256, hidden_size=256))
        self.cell_list = nn.ModuleList(self.cell_list)
        self.pi = nn.Linear(256, self.output_size)
        self.v = nn.Linear(256, 1)

    def forward(self, x, hx, cx, embed):
        output = self.encoder(x)
        output = torch.cat((output, embed), dim=1)
        h_out = []; c_out = []
        for d in range(self.D):
            h, c = self.cell_list[d](output, (hx[d], cx[d]))
            output = h
            h_out.append(h); c_out.append(c)
        return Categorical(logits=self.pi(output)), self.v(output), torch.stack(h_out), torch.stack(c_out)


class ConvGRUPolicy(nn.Module):
    """
    Baseline LSTM Architecture
    """
    def __init__(self, input_size, output_size, use_bn=False,
                cnn_type='nature', D=1, N=1):
        super(ConvGRUPolicy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.use_bn = use_bn
        self.cnn_type = cnn_type
        self.D = D
        self.N = N

        if self.cnn_type == 'nature':
            self.encoder = NatureCnn(input_size=input_size, use_bn=use_bn)
            hidden_size = 512
        elif self.cnn_type == 'impala':
            self.encoder = ImpalaCnn(input_size=input_size, use_bn=use_bn)
            hidden_size = 256

        lstm_input_size = hidden_size + self.output_size + 2
        self.cell_list = [nn.GRUCell(lstm_input_size, hidden_size=256)]
        for d in range(1, self.D):
            self.cell_list.append(nn.GRUCell(256, hidden_size=256))
        self.cell_list = nn.ModuleList(self.cell_list)
        self.pi = nn.Linear(256, self.output_size)
        self.v = nn.Linear(256, 1)

    def forward(self, x, hx, embed):
        output = self.encoder(x)
        output = torch.cat((output, embed), dim=1)
        h_out = []
        for d in range(self.D):
            output = self.cell_list[d](output, hx[d])
            h_out.append(output)
        return Categorical(logits=self.pi(output)), self.v(output), torch.stack(h_out)
