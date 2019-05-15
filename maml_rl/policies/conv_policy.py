import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from collections import OrderedDict
from maml_rl.policies.policy import ImpalaCnn, NatureCnn
import numpy as np

class ConvPolicy(nn.Module):
    """
    Baseline PPO Architecture
    """
    def __init__(self, input_size, output_size, use_bn=False, cnn_type='nature'):
        super(ConvPolicy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.use_bn = use_bn
        self.cnn_type = cnn_type

        if self.cnn_type == 'nature':
            self.encoder = NatureCnn(input_size=input_size, use_bn=use_bn)
            hidden_size = 512
        elif self.cnn_type == 'impala':
            self.encoder = ImpalaCnn(input_size=input_size, use_bn=use_bn)
            hidden_size = 256

        self.pi = nn.Linear(hidden_size, self.output_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output = self.encoder(x)
        return Categorical(logits=self.pi(output)), self.v(output)
