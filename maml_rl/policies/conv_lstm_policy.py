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
                cnn_type='nature', D=1, N=1, device=torch.device('cuda')):
        super(ConvLSTMPolicy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.use_bn = use_bn
        self.cnn_type = cnn_type
        self.D = D
        self.N = N
        self.device = device

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
        self.pi = nn.Linear(256, self.output_size)
        self.v = nn.Linear(256, 1)

    def forward(self, x, hx, cx, embed):
        # state embedding
        output = self.encoder(x)
        # passing joint state + action embedding thru LSTM
        output = torch.cat((output, embed), dim=1)
        # stacked LSTM
        h_out = []; c_out = []
        for d in range(self.D):
            h, c = hx[d], cx[d]
            inner_out = []
            for n in range(self.N):
                if d == 0:
                    h, c = self.cell_list[d](output, (h, c))
                else:
                    h, c = self.cell_list[d](output[n], (h, c))
                inner_out.append(h)
            output = torch.stack(inner_out).float().to(device=self.device)
            h_out.append(h); c_out.append(c)
        h_out = torch.stack(h_out).float().to(device=self.device)
        c_out = torch.stack(c_out).float().to(device=self.device)
        return Categorical(logits=self.pi(h)), self.v(h), h_out, c_out
