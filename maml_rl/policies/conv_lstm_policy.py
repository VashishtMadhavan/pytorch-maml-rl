import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from collections import OrderedDict
from maml_rl.policies.policy import Policy, weight_init

class ConvLSTMPolicy(nn.Module):
    """
    Baseline DQN Architecture
    """
    def __init__(self, input_size, output_size, nonlinearity=F.relu):
        super(ConvPolicy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.nonlinearity = nonlinearity
        self.feature_size = output_size + 2

        self.conv1 = nn.Conv2d(input_size[-1], 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(9 * 9 * 32, 256)
        self.lstm = nn.LSTMCell(256 + self.feature_size, 256)
        self.pi = nn.Linear(256, self.output_size)
        self.v = nn.Linear(256, 1)

    def forward(self, x):
        x, (hx, cx) = x
        featurize = len(x.size()) > 4
        if featurize:
            T, B, H, W, C = x.size()
            output = x.view(T * B, H, W, C)
            output = output.permute(0, 3, 1, 2)
        else:
            output = x.permute(0, 3, 1, 2)

        output = self.nonlinearity(self.conv1(output))
        output = self.nonlinearity(self.conv2(output))
        output = output.view(output.size(0), -1)
        output = self.nonlinearity(self.fc1(output))
        hx, cx = self.lstm(output, (hx, cx))
        output = hx
        logits = self.pi(output)
        values = self.v(output)
        if featurize:
            logits = logits.view(T, B, -1)
            values = values.view(T, B, -1)
        return Categorical(logits=logits), values, (hx, cx)