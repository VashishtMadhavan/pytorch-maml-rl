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
    def __init__(self, input_size, output_size, use_bn=False, cnn_type='nature'):
        super(ConvLSTMPolicy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.use_bn = use_bn
        self.cnn_type = cnn_type

        if self.cnn_type == 'nature':
            self.encoder = NatureCnn(input_size=input_size, use_bn=use_bn)
            hidden_size = 512
        elif self.cnn_type == 'impala':
            self.encoder = ImpalaCnn(input_size=input_size, use_bn=use_bn)
            hidden_size = 256

        self.lstm = nn.LSTMCell(hidden_size + self.output_size + 2, 256)
        self.pi = nn.Linear(256, self.output_size)
        self.v = nn.Linear(256, 1)

    def forward(self, x, hx, cx, embed):
        # state embedding
        output = self.encoder(x)
        # passing joint state + action embedding thru LSTM
        output = torch.cat((output, embed), dim=1)
        h_out, c_out = self.lstm(output, (hx, cx))
        return Categorical(logits=self.pi(h_out)), self.v(h_out), h_out, c_out
