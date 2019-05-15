import torch
import torch.optim as optim
import gym
import numpy as np
from maml_rl.utils.torch_utils import weighted_mean, weighted_normalize
from maml_rl.envs.subproc_vec_env import SubprocVecEnv
from maml_rl.episode import LSTMBatchEpisodes
from maml_rl.policies import ConvCGRUPolicy, ConvGRUPolicy, ConvPolicy
from collections import deque

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
    LSTM Learner using PPO
    """
    def __init__(self, env_name, num_workers, num_batches=1000, n_step=5, gamma=0.95, lr=0.01, 
                tau=1.0, ent_coef=.01, vf_coef=0.5, lstm_size=256, clip_frac=0.2, device='cpu',
                surr_epochs=3, clstm=False, surr_batches=4, max_grad_norm=0.5, cnn_type='nature'):
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.gamma = gamma
        self.use_clstm = clstm
        self.n_step = n_step
        self.reward_log = deque(maxlen=100)
        self.lstm_size = lstm_size

        # Sampler variables
        self.env_name = env_name
        self.num_batches = num_batches
        self.num_workers = num_workers
        self.env_name = env_name
        self.envs = SubprocVecEnv([make_env(env_name) for _ in range(self.num_workers)])
        self.obs_shape = self.envs.observation_space.shape
        self.num_actions = self.envs.action_space.n

        self.obs = np.zeros((self.num_workers, ) + self.obs_shape)
        self.obs[:] = self.envs.reset()
        self.dones = [False for _ in range(self.num_workers)]
        self.embed = torch.zeros(self.num_workers, self.num_actions + 2).to(device=device)
        self.embed[:, 0] = 1.

        if not self.use_clstm:
            self.hx = torch.zeros(self.num_workers, self.lstm_size).to(device=device)
            #self.policy = ConvGRUPolicy(input_size=self.obs_shape, output_size=self.num_actions,
            #    use_bn=False, cnn_type=cnn_type, lstm_size=self.lstm_size)
            self.policy = ConvPolicy(input_size=self.obs_shape, output_size=self.num_actions,
                use_bn=False, cnn_type=cnn_type)
        else:
            self.hx = torch.zeros(self.num_workers, self.lstm_size, 7, 7).to(device=device)
            self.policy = ConvCGRUPolicy(input_size=self.obs_shape, output_size=self.num_actions,
                use_bn=False, cnn_type=cnn_type, lstm_size=self.lstm_size)

        # Optimization Variables
        self.lr = lr
        self.tau = tau
        self.clip_frac = clip_frac
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, eps=1e-5)

        # PPO variables
        self.surrogate_epochs = surr_epochs
        self.surrogate_batches = surr_batches
        self.surrogate_batch_size = self.num_workers // self.surrogate_batches

        self.to(device)
        self.max_grad_norm = max_grad_norm

    def _forward_policy(self, episodes, ratio=False):
        T = episodes.observations.size(0)
        values, log_probs, entropy = [], [], []
        if not self.use_clstm:
            hx = torch.zeros(self.num_workers, self.lstm_size).to(device=self.device)
        else:
            hx = torch.zeros(self.num_workers, self.lstm_size, 7, 7).to(device=self.device)

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

    def loss(self, episodes, inds=None):
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
        for i in range(self.surrogate_epochs):
            for j in range(self.surrogate_batches):
                sample_inds = np.random.choice(self.num_workers, self.surrogate_batch_size, replace=False)
                self.optimizer.zero_grad()
                loss = self.loss(episodes, inds=sample_inds)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def _get_term_flags(self, infos):
        if 'v0' in self.env_name:
            return np.array(self.dones)
        else:
            return np.array([np.sign(v['done']) for k,v in enumerate(infos)])

    def sample(self):
        """
        Sample trajectories
        """
        episodes = LSTMBatchEpisodes(batch_size=self.num_workers, gamma=self.gamma, device=self.device)
        for ns in range(self.n_step):
            with torch.no_grad():
                obs_tensor = torch.from_numpy(self.obs).float().to(device=self.device)
                #act_dist, values_tensor, self.hx = self.policy(obs_tensor, self.hx, self.embed)
                act_dist, values_tensor = self.policy(obs_tensor)
                act_tensor = act_dist.sample()

                # cpu variables for logging
                log_probs = act_dist.log_prob(act_tensor).cpu().numpy()
                actions = act_tensor.cpu().numpy()
                old_values = values_tensor.squeeze().cpu().numpy()
                embed = self.embed.cpu().numpy()
            new_observations, rewards, self.dones, infos = self.envs.step(actions)

            # Update embeddings when episode is done
            term_flags = self._get_term_flags(infos)
            embed_temp = np.hstack((one_hot(actions, self.num_actions), rewards[:, None], term_flags[:, None]))
            self.embed = torch.from_numpy(embed_temp).float().to(device=self.device)

            # Logging episode rew
            for dr in rewards[self.dones == 1]:
                self.reward_log.append(dr)

            # Update hidden states
            dones_tensor = torch.from_numpy(self.dones.astype(np.float32)).to(device=self.device)
            if not self.use_clstm:
                self.hx[:, dones_tensor == 1, :] = 0.
            else:
                self.hx[:, dones_tensor == 1, :, :, :] = 0.

            self.embed[dones_tensor == 1] = 0.
            self.embed[dones_tensor == 1, 0] = 1.

            episodes.append(self.obs, actions, rewards, log_probs, old_values, embed)
            self.obs[:] = new_observations
        return episodes

    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.device = device
