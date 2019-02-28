import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from collections import OrderedDict
from maml_rl.policies.policy import ConvLSTM, ConvGRU
import numpy as np

class ConvCLSTMPolicy(nn.Module):
    """
    Baseline DQN Architecture
    """
    def __init__(self, input_size, output_size, use_bn=False, D=1, N=1):
        super(ConvCLSTMPolicy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.embed_dim = self.output_size + 2
        self.use_bn = use_bn
        self.D = D
        self.N = N

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_size[-1], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.conv_lstm = ConvLSTM(input_size=(7, 7), input_dim=64 + self.embed_dim,
                                hidden_dim=32, kernel_size=3, num_layers=self.D)
        self.fc = nn.Linear(7 * 7 * 32, 256)
        self.pi = nn.Linear(256, self.output_size)
        self.v = nn.Linear(256, 1)

    def forward(self, x, hx, cx, embed):
        # state embedding
        output = x.permute(0, 3, 1, 2)
        output = self.relu(self.conv1(output))
        output = self.relu(self.conv2(output))
        output = self.relu(self.conv3(output))
        
        # tiling the embedding to match shape of the conv tensor
        e_t = embed.unsqueeze(-1).unsqueeze(-1)
        e_t = e_t.expand(e_t.size(0), self.embed_dim, output.size(2), output.size(3))
        output = torch.cat((output, e_t), dim=1)
        output = output.unsqueeze(0).repeat(self.N, 1, 1, 1, 1)

        output, (h_out, c_out) = self.conv_lstm(output, (hx, cx))
        output = output[-1].view(output[-1].size(0), -1)
        output = self.relu(self.fc(output))
        return Categorical(logits=self.pi(output)), self.v(output), h_out, c_out


class ConvCGRUPolicy(nn.Module):
    """
    Baseline DQN Architecture
    """
    def __init__(self, input_size, output_size, use_bn=False, D=1, N=1):
        super(ConvCGRUPolicy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.embed_dim = self.output_size + 2
        self.use_bn = use_bn
        self.D = D
        self.N = N

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_size[-1], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.conv_gru = ConvGRU(input_size=(7, 7), input_dim=64 + self.embed_dim,
                                hidden_dim=32, kernel_size=3, num_layers=self.D)
        self.fc = nn.Linear(7 * 7 * 32, 256)
        self.pi = nn.Linear(256, self.output_size)
        self.v = nn.Linear(256, 1)

    def forward(self, x, hx, embed):
        # state embedding
        output = x.permute(0, 3, 1, 2)
        output = self.relu(self.conv1(output))
        output = self.relu(self.conv2(output))
        output = self.relu(self.conv3(output))

        # tiling the embedding to match shape of the conv tensor
        e_t = embed.unsqueeze(-1).unsqueeze(-1)
        e_t = e_t.expand(e_t.size(0), self.embed_dim, output.size(2), output.size(3))
        output = torch.cat((output, e_t), dim=1)
        output = output.unsqueeze(0).repeat(self.N, 1, 1, 1, 1)

        output, h_out = self.conv_gru(output, hx)
        output = output[-1].view(output[-1].size(0), -1)
        output = self.relu(self.fc(output))
        return Categorical(logits=self.pi(output)), self.v(output), h_out