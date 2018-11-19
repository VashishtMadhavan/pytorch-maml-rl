import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from collections import OrderedDict
from maml_rl.policies.policy import Policy

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

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

        self.lstm = nn.LSTMCell(256 + self.output_size + 2, 256)
        self.pi = nn.Linear(256, self.output_size)
        self.v = nn.Linear(256, 1)

        self.apply(weights_init)
        self.pi.weight.data = normalized_columns_initializer(self.pi.weight.data, 0.01)
        self.pi.bias.data.fill_(0)

        self.v.weight.data = normalized_columns_initializer(self.v.weight.data, 1.0)
        self.v.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

    def forward(self, x, hx, cx, act_embedding, rew_embedding):
        # state embedding
        output = x.permute(0, 3, 1, 2)
        output = self.nonlinearity(self.conv1(output))
        output = self.nonlinearity(self.conv2(output))
        output = output.view(output.size(0), -1)
        output = self.nonlinearity(self.fc(output))

        # passing joint embedding through GRU
        output = torch.cat((output, act_embedding, rew_embedding), dim=1)
        h_out, c_out = self.lstm(output, (hx, cx))
        output = h_out
        return Categorical(logits=self.pi(output)), self.v(output), h_out, c_out