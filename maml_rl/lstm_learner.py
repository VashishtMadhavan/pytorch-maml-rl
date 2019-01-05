import torch
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
from torch.distributions.kl import kl_divergence

from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
                                       weighted_normalize)
from maml_rl.utils.optimization import conjugate_gradient
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

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
    LSTM Learner using A2C/PPO
    """
    def __init__(self, env_name, batch_size, num_workers, num_batches=1000,
                gamma=0.95, lr=0.01, tau=1.0, ent_coef=.01, vf_coef=0.5, lstm_size=256, clip_frac=0.2, device='cpu',
                surr_epochs=4, surr_batches=4, max_grad_norm=0.5):
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.gamma = gamma
        self.device = device
        self.lstm_size = lstm_size

        # Sampler variables
        self.env_name = env_name
        self.num_batches = num_batches
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
        self.clip_frac = clip_frac
        self.optimizer = optim.RMSprop(self.policy.parameters(), lr=self.lr, alpha=0.99, eps=1e-5)

        # PPO variables
        self.surrogate_epochs = surr_epochs
        self.surrogate_batches = surr_batches
        self.surrogate_batch_size = self.batch_size // self.surrogate_batches

        self.to(device)
        self.max_grad_norm = max_grad_norm

    def loss(self, episodes, inds=None):
        """
        REINFORCE gradient with baseline [2], computed on advantages estimated 
        with Generalized Advantage Estimation (GAE, [3]).
        """
        T = episodes.observations.size(0)
        values, log_probs, entropy = [], [], []
        hx = torch.zeros(self.batch_size, self.lstm_size).to(device=self.device)
        cx = torch.zeros(self.batch_size, self.lstm_size).to(device=self.device)
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

        pg_loss = -weighted_mean(log_probs * advantages, dim=0, weights=episodes.mask)
        vf_loss = 0.5 * weighted_mean((values.squeeze() - episodes.returns) ** 2,dim=0, weights=episodes.mask)
        entropy_loss = weighted_mean(entropy, dim=0, weights=episodes.mask)
        return pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy_loss


    def surrogate_loss(self, episodes, inds=None):
        T = episodes.observations.size(0)
        values, log_probs, entropy = [], [], []
        hx = torch.zeros(self.batch_size, self.lstm_size).to(device=self.device)
        cx = torch.zeros(self.batch_size, self.lstm_size).to(device=self.device)
        for t in range(T):
            pi_t, v_t, hx, cx = self.policy(observations[t], hx, cx, episodes.action_embeds[t], episodes.rew_embeds[t])
            values.append(v_t)
            entropy.append(pi_t.entropy())
            log_ratio.append(pi_t.log_prob(episodes.actions[t]) - episodes.logprobs[t])
        log_ratio = torch.stack(log_ratio); values = torch.stack(values); entropy = torch.stack(entropy)

        advantages = episodes.gae(values, tau=self.tau)
        advantages = weighted_normalize(advantages, weights=episodes.mask)

        if log_ratio.dim() > 2:
            log_ratio = torch.sum(log_ratio, dim=2)

        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, min=1.0 - self.clip_frac, max=1.0 + self.clip_frac)

        # TODO: potentially restrict value func deviation as well
        if not inds:
            pg_loss = weighted_mean(torch.max(pg_loss1, pg_loss2), dim=0, weights=episodes.mask)
            vf_loss = 0.5 * weighted_mean((values.squeeze() - episodes.returns) ** 2, dim=0, weights=episodes.mask)
            entropy_loss = weighted_mean(entropy, dim=0, weights=episodes.mask)
        else:
            masks = episodes.mask[:, ind]
            pg_loss = weighted_mean(torch.max(pg_loss1, pg_loss2)[:, ind], dim=0, weights=masks)
            vf_loss = 0.5 * weighted_mean((values.squeeze()[:, inds] - episodes.returns[:, inds]) ** 2, dim=0, weights=masks)
            entropy_loss = weighted_mean(entropy[:, inds], dim=0, weights=masks)
        return pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy_loss


    def step(self, episodes):
        """
        Adapt the parameters of the policy network to a new set of examples
        """
        self.optimizer.zero_grad()
        loss = self.loss(episodes)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def surrogate_step(self, episodes):
        for se in range(self.surrogate_epochs):
            for k in range(self.surrogate_batches):
                sample_inds = np.random.choice(self.batch_size, self.surrogate_batch_size, replace=False)
                self.optimizer.zero_grad()
                loss = self.surrogate_loss(episodes, inds=sample_inds)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def sample(self):
        """
        Sample trajectories
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
        hx = torch.zeros(self.num_workers, self.lstm_size).to(device=self.device)
        cx = torch.zeros(self.num_workers, self.lstm_size).to(device=self.device)

        while (not all(dones)) or (not self.queue.empty()):
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).to(device=self.device)
                actions_dist, values_tensor, hx, cx = self.policy(observations_tensor, hx, cx, action_embed_tensor, rew_embed_tensor)
                actions = actions_dist.sample()
                log_probs = actions_dist.log_prob(actions).cpu().numpy()
                actions = actions.cpu().numpy()
                action_embed = action_embed_tensor.cpu().numpy()
                rew_embed = rew_embed_tensor.cpu().numpy()
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)

            # Update embeddings when episode is done
            actions_mask = ((1. - dones.astype(np.float32)) * actions).astype(np.int32)
            action_embed_tensor = torch.from_numpy(one_hot(actions_mask, num_actions)).float().to(device=self.device)
            rew_embed_tensor = torch.from_numpy(np.hstack((rewards[:, None], dones[:, None]))).float().to(device=self.device)

            # Update hidden states
            dones_tensor = torch.from_numpy(dones.astype(np.float32)).to(device=self.device)
            hx[dones_tensor == 1] = 0.
            cx[dones_tensor == 1] = 0.
            rew_embed_tensor[dones_tensor == 1] == 0.

            episodes.append(observations, actions, rewards, batch_ids, log_probs, action_embed, rew_embed)
            observations, batch_ids = new_observations, new_batch_ids
        return episodes

    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.device = device
