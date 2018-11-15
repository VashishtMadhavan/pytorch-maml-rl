import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from collections import OrderedDict
from maml_rl.policies.policy import Policy, weight_init

class ConvPolicy(Policy):
    """
    Baseline DQN Architecture
    """
    def __init__(self, input_size, output_size, nonlinearity=F.relu):
        super(ConvPolicy, self).__init__(input_size=input_size, output_size=output_size)
        self.nonlinearity = nonlinearity

        self.add_module('conv1', nn.Conv2d(input_size[-1], 16, kernel_size=8, stride=4))
        self.add_module('conv2', nn.Conv2d(16, 32, kernel_size=4, stride=2))
        self.add_module('fc1', nn.Linear(9 * 9 * 32, 256))
        self.add_module('out', nn.Linear(256, output_size))
        self.apply(weight_init)

    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        featurize = len(x.size()) > 4
        if featurize:
            T, B, H, W, C = x.size()
            output = x.view(T * B, H, W, C)
            output = output.permute(0, 3, 1, 2)
        else:
            output = x.permute(0, 3, 1, 2)
        output = F.conv2d(output, weight=params['conv1.weight'], bias=params['conv1.bias'], stride=4)
        output = self.nonlinearity(output)
        output = F.conv2d(output, weight=params['conv2.weight'], bias=params['conv2.bias'], stride=2)
        output = self.nonlinearity(output)
        output = output.view(output.size(0), -1)
        output = F.linear(output, weight=params['fc1.weight'], bias=params['fc1.bias'])
        output = self.nonlinearity(output)
        logits = F.linear(output, weight=params['out.weight'], bias=params['out.bias'])
        if featurize:
            logits = logits.view(T, B, -1)
        return Categorical(logits=logits)
