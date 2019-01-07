import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from collections import OrderedDict
from maml_rl.policies.policy import Policy
import numpy as np

class ConvLSTMPolicy(nn.Module):
    """
    Baseline DQN Architecture
    """
    def __init__(self, input_size, output_size, nonlinearity=F.relu):
        super(ConvLSTMPolicy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.nonlinearity = nonlinearity

        self.conv1 = nn.Conv2d(input_size[-1], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(7 * 7 * 64, 512)

        self.lstm = nn.LSTMCell(512 + self.output_size + 2, 256)
        self.pi = nn.Linear(256, self.output_size)
        self.v = nn.Linear(256, 1)

    def forward(self, x, hx, cx, embed):
        # state embedding
        output = x.permute(0, 3, 1, 2)
        output = self.nonlinearity(self.conv1(output))
        output = self.nonlinearity(self.conv2(output))
        output = self.nonlinearity(self.conv3(output))
        output = output.view(output.size(0), -1)
        output = self.nonlinearity(self.fc(output))

        # passing joint state + action embedding thru LSTM
        output = torch.cat((output, embed), dim=1)
        h_out, c_out = self.lstm(output, (hx, cx))
        output = h_out
        return Categorical(logits=self.pi(output)), self.v(output), h_out, c_out