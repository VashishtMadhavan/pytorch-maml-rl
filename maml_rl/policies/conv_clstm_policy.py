import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from collections import OrderedDict
from maml_rl.policies.policy import ConvLSTMCell
import numpy as np

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