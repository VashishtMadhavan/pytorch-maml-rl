import torch
from maml_rl.utils.torch_utils import (weighted_mean, weighted_normalize)
import torch.optim as optim
import gym
import numpy as np
import multiprocessing as mp
from maml_rl.envs.subproc_vec_env import SubprocVecEnv
from maml_rl.episode import LSTMBatchEpisodes
from maml_rl.policies import LSTMPolicy, GRUPolicy

def make_env(env_name):
    def _make_env():
        return gym.make(env_name)
    return _make_env

def one_hot(actions, num_actions):
    x = np.zeros((len(actions), num_actions))
    x[np.arange(len(actions)), actions] = 1.
    return x

class BanditLearner(object):
    """
    LSTM Learner using A2C/PPO
    """
    def __init__(self, k, n, batch_size, num_workers, num_batches=1000, gamma=0.95, lr=0.01, 
    	        tau=1.0, ent_coef=.01, vf_coef=0.5, lstm_size=50, clip_frac=0.2, device='cpu',
                surr_epochs=3, surr_batches=4, max_grad_norm=0.5, D=1):
        self.k = k
        self.n = n
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.gamma = gamma
        self.D = D

        # Sampler variables
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.queue = mp.Queue()
        self.env_name = 'Bandit-K{0}-N{1}-v0'.format(self.k, self.n)
        self.envs = SubprocVecEnv([make_env(self.env_name) for _ in range(num_workers)], queue=self.queue)
        self.obs_shape = self.envs.observation_space.shape
        self.num_actions = self.envs.action_space.n

        self.lstm_size = lstm_size
        self.policy = GRUPolicy(input_size=self.obs_shape[0], output_size=self.num_actions, lstm_size=self.lstm_size, D=self.D)

        # Optimization Variables
        self.lr = lr
        self.tau = tau
        self.clip_frac = clip_frac
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, eps=1e-5)

        # PPO variables
        self.surrogate_epochs = surr_epochs
        self.surrogate_batches = surr_batches
        self.surrogate_batch_size = self.batch_size // self.surrogate_batches

        self.to(device)
        self.max_grad_norm = max_grad_norm

    def _forward_policy(self, episodes, ratio=False):
        T = episodes.observations.size(0)
        values, log_probs, entropy = [], [], []
        hx = torch.zeros(self.D, self.batch_size, self.lstm_size).to(device=self.device)
        
        for t in range(T):
            pi, v, hx = self.policy(episodes.observations[t], hx, episodes.embeds[t])
            values.append(v)
            entropy.append(pi.entropy())
            if ratio:
                log_probs.append(pi.log_prob(episodes.actions[t]) - episodes.logprobs[t])
            else:
                log_probs.append(pi.log_prob(episodes.actions[t]))

        log_probs = torch.stack(log_probs); values = torch.stack(values); entropy = torch.stack(entropy)
        advantages = episodes.gae(values, tau=self.tau)
        advantages = weighted_normalize(advantages, weights=episodes.mask)
        if log_probs.dim() > 2:
            log_probs = torch.sum(log_probs, dim=2)

        return log_probs, advantages, values, entropy

    def loss(self, episodes):
        """
        REINFORCE gradient with baseline [2], computed on advantages estimated 
        with Generalized Advantage Estimation (GAE, [3]).
        """
        log_probs, advantages, values, entropy = self._forward_policy(episodes)

        pg_loss = -weighted_mean(log_probs * advantages, dim=0, weights=episodes.mask)
        vf_loss = 0.5 * weighted_mean((values.squeeze() - episodes.returns) ** 2, dim=0, weights=episodes.mask)
        entropy_loss = weighted_mean(entropy, dim=0, weights=episodes.mask)
        return pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy_loss


    def surrogate_loss(self, episodes, inds=None):
        """
        PPO Surrogate Loss
        """
        log_ratios, advantages, values, entropy = self._forward_policy(episodes, ratio=True)

        # clipped pg loss
        ratio = torch.exp(log_ratios)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, min=1.0 - self.clip_frac, max=1.0 + self.clip_frac)

        # clipped value loss
        values_clipped = episodes.old_values + torch.clamp(values.squeeze() - episodes.old_values, min=-self.clip_frac, max=self.clip_frac)
        vf_loss1 = (values.squeeze() - episodes.returns) ** 2
        vf_loss2 = (values_clipped - episodes.returns) ** 2

        if inds is None:
            inds = np.arange(self.batch_size)

        masks = episodes.mask[:, inds]
        pg_loss = weighted_mean(torch.max(pg_loss1, pg_loss2)[:, inds], dim=0, weights=masks)
        vf_loss = 0.5 * weighted_mean(torch.max(vf_loss1, vf_loss2)[:, inds], dim=0, weights=masks)
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
        for _ in range(self.surrogate_epochs):
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

        observations, batch_ids = self.envs.reset()
        dones = [False]; timers = np.zeros(self.num_workers)

        embed_tensor = torch.zeros(self.num_workers, self.num_actions + 3).to(device=self.device)
        embed_tensor[:, 0] = 1.
        hx = torch.zeros(self.D, self.num_workers, self.lstm_size).to(device=self.device)

        while (not all(dones)) or (not self.queue.empty()):
            with torch.no_grad():
                obs_tensor = torch.from_numpy(observations).to(device=self.device)
                act_dist, values_tensor, hx = self.policy(obs_tensor, hx, embed_tensor)
                act_tensor = act_dist.sample()

                # cpu variables for logging
                log_probs = act_dist.log_prob(act_tensor).cpu().numpy()
                actions = act_tensor.cpu().numpy()
                old_values = values_tensor.squeeze().cpu().numpy()
                embed = embed_tensor.cpu().numpy()
            new_observations, rewards, dones, new_batch_ids, infos = self.envs.step(actions)
            timers += 1.0

            # Update embeddings when episode is done
            embed_temp = np.hstack((one_hot(actions, self.num_actions), rewards[:, None], dones[:, None], timers[:, None]))
            embed_tensor = torch.from_numpy(embed_temp).float().to(device=self.device)

            # Update hidden states
            dones_tensor = torch.from_numpy(dones.astype(np.float32)).to(device=self.device)
            timers[dones] = 0.
            hx[:, dones_tensor == 1, :] = 0.
            embed_tensor[dones_tensor == 1] = 0.
            embed_tensor[dones_tensor == 1, 0] = 1.

            episodes.append(observations, actions, rewards, batch_ids, log_probs, old_values, embed)
            observations, batch_ids = new_observations, new_batch_ids
        return episodes

    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.device = device
