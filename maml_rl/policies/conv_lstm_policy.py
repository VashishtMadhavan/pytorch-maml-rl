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
        super(ConvLSTMPolicy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.nonlinearity = nonlinearity

        self.conv1 = nn.Conv2d(input_size[-1], 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc = nn.Linear(9 * 9 * 32, 256)

        self.gru = nn.GRUCell(256 + self.output_size + 2, 256)
        self.pi = nn.Linear(256, self.output_size)
        self.v = nn.Linear(256, 1)

    def forward(self, x, hx, act_embedding, rew_embedding):
        # state embedding
        output = x.permute(0, 3, 1, 2)
        output = self.nonlinearity(self.conv1(output))
        output = self.nonlinearity(self.conv2(output))
        output = output.view(output.size(0), -1)
        output = self.nonlinearity(self.fc(output))

        # passing joint embedding through GRU
        output = torch.cat((output, act_embedding, rew_embedding), dim=1)
        h_out = self.gru(output, hx)
        output = h_out
        return Categorical(logits=self.pi(output)), self.v(output), h_out