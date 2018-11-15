import torch
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
from torch.distributions.kl import kl_divergence

from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
                                       weighted_normalize)
from maml_rl.utils.optimization import conjugate_gradient
import torch.optim as optim

class LSTMLearner(object):
    """
    LSTM Learner using A2C to Train
    """
    def __init__(self, sampler, policy, gamma=0.95,
                 lr=0.5, tau=1.0, vf_coef=0.5, device='cpu'):
        self.sampler = sampler
        self.policy = policy
        self.vf_coef = vf_coef
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.to(device)

    def loss(self, episodes):
        """Compute the inner loss for the one-step gradient update. The inner 
        loss is REINFORCE with baseline [2], computed on advantages estimated 
        with Generalized Advantage Estimation (GAE, [3]).
        """
        pi, values = self.policy((episodes.observations, (hx, cx)))
        advantages = episodes.gae(values, tau=self.tau)
        advantages = weighted_normalize(advantages, weights=episodes.mask)

        log_probs = pi.log_prob(episodes.actions)
        if log_probs.dim() > 2:
            log_probs = torch.sum(log_probs, dim=2)
        loss = -weighted_mean(log_probs * advantages, dim=0,
            weights=episodes.mask)
        vf_loss = 0.5 * weighted_mean((values.squeeze() - episodes.returns) ** 2,
            dim=0, weights=episodes.mask)
        return loss + self.vf_coef * vf_loss

    def step(self, episodes):
        """
        Adapt the parameters of the policy network to a new set of examples
        """
        self.optimizer.zero_grad()
        loss = self.loss(episodes)
        loss.backward()
        self.optimizer.step()

    def sample(self):
        """
        Sample trajectories (before and after the update of the parameters) 
        for all the tasks `tasks`.
        """
        train_episodes = self.sampler.sample(self.policy, gamma=self.gamma, device=self.device)
        return train_episodes

    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.device = device
