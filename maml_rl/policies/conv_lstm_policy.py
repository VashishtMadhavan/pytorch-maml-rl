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
                cnn_type='nature', lstm_size=256):
        super(ConvLSTMPolicy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.use_bn = use_bn
        self.cnn_type = cnn_type
        self.lstm_size = lstm_size

        if self.cnn_type == 'nature':
            self.encoder = NatureCnn(input_size=input_size, use_bn=use_bn)
            hidden_size = 512
        elif self.cnn_type == 'impala':
            self.encoder = ImpalaCnn(input_size=input_size, use_bn=use_bn)
            hidden_size = 256

        lstm_input_size = hidden_size + self.output_size + 2
        self.lstm = nn.LSTMCell(lstm_input_size, hidden_size=self.lstm_size)
        self.pi = nn.Linear(self.lstm_size, self.output_size)
        self.v = nn.Linear(self.lstm_size, 1)

    def forward(self, x, hx, cx, embed):
        output = self.encoder(x)
        output = torch.cat((output, embed), dim=1)
        h, c = self.lstm(output, (hx, cx))
        return Categorical(logits=self.pi(h)), self.v(h), h, c

class ConvGRUPolicy(nn.Module):
    """
    Baseline GRU Architecture
    """
    def __init__(self, input_size, output_size, use_bn=False,
                cnn_type='nature', lstm_size=256):
        super(ConvGRUPolicy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.use_bn = use_bn
        self.cnn_type = cnn_type
        self.lstm_size = lstm_size

        if self.cnn_type == 'nature':
            self.encoder = NatureCnn(input_size=input_size, use_bn=use_bn)
            hidden_size = 512
        elif self.cnn_type == 'impala':
            self.encoder = ImpalaCnn(input_size=input_size, use_bn=use_bn)
            hidden_size = 256

        lstm_input_size = hidden_size + self.output_size + 2
        self.gru = nn.GRUCell(lstm_input_size, hidden_size=self.lstm_size)
        self.pi = nn.Linear(self.lstm_size, self.output_size)
        self.v = nn.Linear(self.lstm_size, 1)

    def forward(self, x, hx, embed):
        output = self.encoder(x)
        output = torch.cat((output, embed), dim=1)
        h = self.gru(output, hx)
        return Categorical(logits=self.pi(h)), self.v(h), h
