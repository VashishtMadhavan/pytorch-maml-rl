import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from collections import OrderedDict
from maml_rl.policies.policy import Policy
import numpy as np


class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias=True):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()
        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, x, state):
        hx, cx = state
        comb = torch.cat([x, hx], dim=1)  # concatenate along channel axis
        
        comb_conv = self.conv(comb)
        cc_i, cc_f, cc_o, cc_g = torch.split(comb_conv, self.hidden_dim, dim=1)
        cx = torch.sigmoid(cc_f) * cx + torch.sigmoid(cc_i) * torch.tanh(cc_g)
        hx = torch.sigmoid(cc_o) * torch.tanh(cx)
        return hx, cx

class ConvCLSTMPolicy(nn.Module):
    """
    Baseline DQN Architecture
    """
    def __init__(self, input_size, output_size, nonlinearity=F.relu, use_bn=False):
        super(ConvCLSTMPolicy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.nonlinearity = nonlinearity
        self.embed_dim = self.output_size + 2
        self.use_bn = use_bn

        self.conv1 = nn.Conv2d(input_size[-1], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.conv_lstm = ConvLSTMCell(input_size=(7, 7), input_dim=64 + self.embed_dim,
                                    hidden_dim=32, kernel_size=3)

        self.fc = nn.Linear(7 * 7 * 32, 256)
        self.pi = nn.Linear(256, self.output_size)
        self.v = nn.Linear(256, 1)

    def forward(self, x, hx, cx, embed):
        # state embedding
        output = x.permute(0, 3, 1, 2)
        output = self.nonlinearity(self.conv1(output))
        output = self.nonlinearity(self.conv2(output))
        output = self.nonlinearity(self.conv3(output))
        
        # tiling the embedding to match shape of the conv tensor
        e_t = embed.unsqueeze(-1).unsqueeze(-1)
        e_t = e_t.expand(e_t.size(0), self.embed_dim, output.size(2), output.size(3))
        output = torch.cat((output, e_t), dim=1)

        h_out, c_out = self.conv_lstm(output, (hx, cx))
        output = h_out.view(h_out.size(0), -1)
        output = self.nonlinearity(self.fc(output))
        return Categorical(logits=self.pi(output)), self.v(output), h_out, c_out