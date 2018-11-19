import torch
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
from torch.distributions.kl import kl_divergence

from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
                                       weighted_normalize)
from maml_rl.utils.optimization import conjugate_gradient
import torch.optim as optim

import gym
import multiprocessing as mp
import numpy as np

from maml_rl.envs.subproc_vec_env import SubprocVecEnv
from maml_rl.episode import LSTMBatchEpisodes
from maml_rl.policies import ConvLSTMPolicy

def make_env(env_name):
    def _make_env():
        return gym.make(env_name)
    return _make_env

def one_hot(actions, num_actions):
    x = np.zeros((len(actions), num_actions))
    x[np.arange(len(actions)), actions] = 1.
    return x

class LSTMLearner(object):
    """
    LSTM Learner using A2C to Train
    """
    def __init__(self, env_name, batch_size, num_workers,
                gamma=0.95, lr=0.01, tau=1.0, ent_coef=.01, vf_coef=0.5, device='cpu',
                max_grad_norm=0.5):
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.gamma = gamma
        self.device = device

        # Sampler variables
        self.env_name = env_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.queue = mp.Queue()
        self.envs = SubprocVecEnv([make_env(env_name) for _ in range(num_workers)], queue=self.queue)
        self._env = gym.make(env_name)

        self.policy = ConvLSTMPolicy(input_size=self.envs.observation_space.shape,
                                     output_size=self.envs.action_space.n)

        # Optimization Variables
        self.lr = lr
        self.tau = tau
        self.optimizer = optim.RMSprop(self.policy.parameters(), lr=self.lr, alpha=0.99, epsilon=1e-5)
        self.to(device)
        self.max_grad_norm = max_grad_norm

    def loss(self, episodes):
        """Compute the inner loss for the one-step gradient update. The inner 
        loss is REINFORCE with baseline [2], computed on advantages estimated 
        with Generalized Advantage Estimation (GAE, [3]).
        """
        T = episodes.observations.size(0)
        values, log_probs, entropy = [], [], []
        hx = torch.zeros(self.batch_size, 256).to(device=self.device)
        cx = torch.zeros(self.batch_size, 256).to(device=self.device)
        for t in range(T):
            pi_t, v_t, hx, cx = self.policy(episodes.observations[t], hx, cx, episodes.action_embeds[t], episodes.rew_embeds[t])
            values.append(v_t)
            entropy.append(pi_t.entropy())
            log_probs.append(pi_t.log_prob(episodes.actions[t]))

        log_probs = torch.stack(log_probs); values = torch.stack(values); entropy = torch.stack(entropy)
        advantages = episodes.gae(values, tau=self.tau)
        advantages = weighted_normalize(advantages, weights=episodes.mask)

        if log_probs.dim() > 2:
            log_probs = torch.sum(log_probs, dim=2)
        loss = -weighted_mean(log_probs * advantages, dim=0,
            weights=episodes.mask)
        vf_loss = 0.5 * weighted_mean((values.squeeze() - episodes.returns) ** 2,
            dim=0, weights=episodes.mask)
        entropy_loss = weighted_mean(entropy, dim=0, weights=episodes.mask)
        return loss + self.vf_coef * vf_loss - self.ent_coef * entropy_loss

    def step(self, episodes):
        """
        Adapt the parameters of the policy network to a new set of examples
        """
        self.optimizer.zero_grad()
        loss = self.loss(episodes)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def sample(self):
        """
        Sample trajectories (before and after the update of the parameters) 
        for all the tasks `tasks`.
        """
        episodes = LSTMBatchEpisodes(batch_size=self.batch_size, gamma=self.gamma, device=self.device)
        for i in range(self.batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)

        self.envs.reset_task([None for _ in range(self.num_workers)])
        observations, batch_ids = self.envs.reset()
        dones = [False]; num_actions = self.envs.action_space.n

        action_embed_tensor = torch.zeros(self.num_workers, num_actions).to(device=self.device)
        action_embed_tensor[:, 0] = 1.
        rew_embed_tensor = torch.zeros(self.num_workers, 2).to(device=self.device)
        hx = torch.zeros(self.num_workers, 256).to(device=self.device)
        cx = torch.zeros(self.num_workers, 256).to(device=self.device)

        while (not all(dones)) or (not self.queue.empty()):
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).to(device=self.device)
                actions_dist, values_tensor, hx, cx = self.policy(observations_tensor, hx, cx, action_embed_tensor, rew_embed_tensor)
                actions = actions_dist.sample().cpu().numpy()
                action_embed = action_embed_tensor.cpu().numpy()
                rew_embed = rew_embed_tensor.cpu().numpy()
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)

            # update embeddings
            # this basically sets the action embedding to the 0 embedding if done
            actions_mask = ((1. - dones.astype(np.float32)) * actions).astype(np.int32)
            action_embed_tensor = torch.from_numpy(one_hot(actions_mask, num_actions)).float().to(device=self.device)
            rew_embed_tensor = torch.from_numpy(np.hstack((rewards[:, None], dones[:, None]))).float().to(device=self.device)

            # update hidden states
            dones_tensor = torch.from_numpy(dones.astype(np.float32)).to(device=self.device)
            hx[dones_tensor == 1] = 0.
            cx[dones_tensor == 1] = 0.
            rew_embed_tensor[dones_tensor == 1] == 0.

            episodes.append(observations, actions, rewards, batch_ids, action_embed, rew_embed)
            observations, batch_ids = new_observations, new_batch_ids
        return episodes

    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.device = device
