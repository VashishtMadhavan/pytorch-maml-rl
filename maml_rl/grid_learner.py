import torch
from maml_rl.utils.torch_utils import (weighted_mean, weighted_normalize)
import torch.optim as optim
import gym
import numpy as np
from maml_rl.envs.subproc_vec_env import SubprocVecEnv
from maml_rl.episode import LSTMBatchEpisodes
from maml_rl.policies import FFPolicy, GRUPolicy
from collections import deque

def make_env(env_name):
    def _make_env():
        return gym.make(env_name)
    return _make_env

def one_hot(actions, num_actions):
    x = np.zeros((len(actions), num_actions))
    x[np.arange(len(actions)), actions] = 1.
    return x

class GridLearner(object):
    """
    GRU Learner using A2C/PPO
    """
    def __init__(self, env_name, num_workers, num_batches=1000, gamma=0.95, lr=0.01, 
                tau=1.0, ent_coef=.01, vf_coef=0.5, lstm_size=32, clip_frac=0.2, device='cpu',
                surr_epochs=3, surr_batches=4, max_grad_norm=0.5, D=1, N=1, n_step=5):
        self.env_name = env_name
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.gamma = gamma
        self.D = D
        self.N = N
        self.n_step = n_step
        self.lstm_size = lstm_size
        self.reward_log = deque(maxlen=100)

        # Sampler variables
        self.num_batches = num_batches
        self.num_workers = num_workers
        self.envs = SubprocVecEnv([make_env(self.env_name) for _ in range(num_workers)])
        self.obs_shape = self.envs.observation_space.shape
        self.num_actions = self.envs.action_space.n

        self.obs = np.zeros((num_workers, ) + self.obs_shape)
        self.obs[:] = self.envs.reset()
        self.dones = [False for _ in range(num_workers)]
        self.hx = torch.zeros(self.D, self.num_workers, self.lstm_size, 2, 2).to(device=device)
        self.embed = torch.zeros(self.num_workers, self.num_actions + 2).to(device=device)
        self.embed[:, 0] = 1.

        #self.policy = GRUPolicy(input_size=self.obs_shape[0], output_size=self.num_actions, lstm_size=self.lstm_size, D=self.D, N=self.N)
        self.policy = FFPolicy(input_size=self.obs_shape[0], output_size=self.num_actions, D=self.D)

        # Optimization Variables
        self.lr = lr
        self.tau = tau
        self.clip_frac = clip_frac
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, eps=1e-4)

        # PPO variables
        self.surrogate_epochs = surr_epochs
        self.surrogate_batches = surr_batches
        self.surrogate_batch_size = self.num_workers // self.surrogate_batches

        self.to(device)
        self.max_grad_norm = max_grad_norm

    def _forward_policy(self, episodes, ratio=False):
        T = episodes.observations.size(0)
        values, log_probs, entropy = [], [], []
        hx = torch.zeros(self.D, self.num_workers, self.lstm_size, 2, 2).to(device=self.device)
        
        for t in range(T):
            #pi, v, hx = self.policy(episodes.observations[t], hx, episodes.embeds[t])
            pi, v = self.policy(episodes.observations[t])
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
            inds = np.arange(self.num_workers)

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
                sample_inds = np.random.choice(self.num_workers, self.surrogate_batch_size, replace=False)
                self.optimizer.zero_grad()
                loss = self.surrogate_loss(episodes, inds=sample_inds)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def sample(self):
        """
        Sample trajectories
        """
        episodes = LSTMBatchEpisodes(batch_size=self.num_workers, gamma=self.gamma, device=self.device)
        for ns in range(self.n_step):
            with torch.no_grad():
                obs_tensor = torch.from_numpy(self.obs).float().to(device=self.device)
                #act_dist, values_tensor, self.hx = self.policy(obs_tensor, self.hx, embed_tensor)
                act_dist, values_tensor = self.policy(obs_tensor)
                act_tensor = act_dist.sample()

                # cpu variables for logging
                log_probs = act_dist.log_prob(act_tensor).cpu().numpy()
                actions = act_tensor.cpu().numpy()
                old_values = values_tensor.squeeze().cpu().numpy()
                embed = self.embed.cpu().numpy()
            new_observations, rewards, self.dones, infos = self.envs.step(actions)

            # logging episode rew
            for dr in rewards[self.dones]:
                self.reward_log.append(dr)

            # Update embeddings when episode is done
            embed_temp = np.hstack((one_hot(actions, self.num_actions), rewards[:, None], self.dones[:, None]))
            self.embed = torch.from_numpy(embed_temp).float().to(device=self.device)

            # Update hidden states
            dones_tensor = torch.from_numpy(self.dones.astype(np.float32)).to(device=self.device)
            #self.hx[:, dones_tensor == 1, :] = 0.
            self.hx[:, dones_tensor == 1, :, :, :] = 0.
            self.embed[dones_tensor == 1] = 0.
            self.embed[dones_tensor == 1, 0] = 1.

            episodes.append(self.obs, actions, rewards, log_probs, old_values, embed)
            self.obs[:] = new_observations
        return episodes

    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.device = device
