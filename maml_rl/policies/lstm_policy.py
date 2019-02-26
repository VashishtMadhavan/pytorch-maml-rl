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
    def __init__(self, input_size, enc_model, output_size, device=torch.device('cuda')):
        super(LSTMPolicy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.encoder = enc_model
        self.D = 1; self.N = 1

        lstm_input_size = self.input_size + self.output_size + 2
        self.cell_list = [nn.LSTMCell(lstm_input_size, hidden_size=256)]
        self.cell_list = nn.ModuleList(self.cell_list)
        self.pi = nn.Linear(256, self.output_size)
        self.v = nn.Linear(256, 1)

    def forward(self, x, hx, cx, embed):
        # state embedding
        output = x.permute(0, 3, 1, 2)
        mu_out, sig_out = self.encoder.encode(output)
        output = self.encoder.sample_latent(mu_out, sig_out)
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
            output = torch.stack(inner_out)
            h_out.append(h); c_out.append(c)
        return Categorical(logits=self.pi(h)), self.v(h), torch.stack(h_out), torch.stack(c_out)
