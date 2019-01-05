import numpy as np
import torch
import torch.nn.functional as F

class BatchEpisodes(object):
    def __init__(self, batch_size, gamma=0.95, device='cpu'):
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device

        self._observations_list = [[] for _ in range(batch_size)]
        self._actions_list = [[] for _ in range(batch_size)]
        self._rewards_list = [[] for _ in range(batch_size)]
        self._mask_list = []

        self._observations = None
        self._actions = None
        self._rewards = None
        self._returns = None
        self._mask = None

    @property
    def observations(self):
        if self._observations is None:
            observation_shape = self._observations_list[0][0].shape
            observations = np.zeros((len(self), self.batch_size)
                + observation_shape, dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._observations_list[i])
                observations[:length, i] = np.stack(self._observations_list[i], axis=0)
            self._observations = torch.from_numpy(observations).to(self.device)
        return self._observations

    @property
    def actions(self):
        if self._actions is None:
            action_shape = self._actions_list[0][0].shape
            actions = np.zeros((len(self), self.batch_size)
                + action_shape, dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._actions_list[i])
                actions[:length, i] = np.stack(self._actions_list[i], axis=0)
            self._actions = torch.from_numpy(actions).to(self.device)
        return self._actions

    @property
    def rewards(self):
        if self._rewards is None:
            rewards = np.zeros((len(self), self.batch_size), dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._rewards_list[i])
                rewards[:length, i] = np.stack(self._rewards_list[i], axis=0)
            self._rewards = torch.from_numpy(rewards).to(self.device)
        return self._rewards

    @property
    def returns(self):
        if self._returns is None:
            return_ = np.zeros(self.batch_size, dtype=np.float32)
            returns = np.zeros((len(self), self.batch_size), dtype=np.float32)
            rewards = self.rewards.cpu().numpy()
            mask = self.mask.cpu().numpy()
            for i in range(len(self) - 1, -1, -1):
                return_ = self.gamma * return_ + rewards[i] * mask[i]
                returns[i] = return_
            self._returns = torch.from_numpy(returns).to(self.device)
        return self._returns

    @property
    def mask(self):
        if self._mask is None:
            mask = np.zeros((len(self), self.batch_size), dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._actions_list[i])
                mask[:length, i] = 1.0
            self._mask = torch.from_numpy(mask).to(self.device)
        return self._mask

    def gae(self, values, tau=1.0):
        # Add an additional 0 at the end of values for
        # the estimation at the end of the episode
        values = values.squeeze(2).detach()
        values = F.pad(values * self.mask, (0, 0, 0, 1))

        deltas = self.rewards + self.gamma * values[1:] - values[:-1]
        advantages = torch.zeros_like(deltas).float()
        gae = torch.zeros_like(deltas[0]).float()
        for i in range(len(self) - 1, -1, -1):
            gae = gae * self.gamma * tau + deltas[i]
            advantages[i] = gae

        return advantages

    def append(self, observations, actions, rewards, batch_ids):
        for observation, action, reward, batch_id in zip(
                observations, actions, rewards, batch_ids):
            if batch_id is None:
                continue
            self._observations_list[batch_id].append(observation.astype(np.float32))
            self._actions_list[batch_id].append(action.astype(np.float32))
            self._rewards_list[batch_id].append(reward.astype(np.float32))

    def __len__(self):
        return max(map(len, self._rewards_list))


class LSTMBatchEpisodes(BatchEpisodes):
    def __init__(self, batch_size, gamma=0.95, device='cpu'):
        super(LSTMBatchEpisodes, self).__init__(batch_size=batch_size, gamma=gamma, device=device)
        self._action_embed_list = [[] for _ in range(batch_size)]
        self._rew_embed_list = [[] for _ in range(batch_size)]
        self._logprob_list = [[] for _ in range(batch_size)]
        self._old_value_list = [[] for _ in range(batch_size)]

        self._action_embed = None
        self._rew_embed = None
        self._logprob = None
        self._old_value = None

    @property
    def action_embeds(self):
        if self._action_embed is None:
            act_embed_shape = self._action_embed_list[0][0].shape
            act_embeds = np.zeros((len(self), self.batch_size) + act_embed_shape, dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._action_embed_list[i])
                act_embeds[:length, i] = np.stack(self._action_embed_list[i], axis=0)
            self._action_embed = torch.from_numpy(act_embeds).to(self.device)
        return self._action_embed

    @property
    def rew_embeds(self):
        if self._rew_embed is None:
            rew_embed_shape = self._rew_embed_list[0][0].shape
            rew_embeds = np.zeros((len(self), self.batch_size) + rew_embed_shape, dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._rew_embed_list[i])
                rew_embeds[:length, i] = np.stack(self._rew_embed_list[i], axis=0)
            self._rew_embed = torch.from_numpy(rew_embeds).to(self.device)
        return self._rew_embed

    @property
    def logprobs(self):
        if self._logprob is None:
            logprobs = np.zeros((len(self), self.batch_size), dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._logprob_list[i])
                logprobs[:length, i] = np.stack(self._logprob_list[i], axis=0)
            self._logprob = torch.from_numpy(logprobs).to(self.device)
        return self._logprob

    @property
    def old_values(self):
        if self._old_value is None:
            values = np.zeros((len(self), self.batch_size), dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self._old_value_list[i])
                values[:length, i] = np.stack(self._old_value_list[i], axis=0)
            self._old_value = torch.from_numpy(values).to(self.device)
        return self._old_value



    def append(self, observations, actions, rewards, batch_ids, log_probs, old_values, action_embeds, rew_embeds):
        for observation, action, reward, batch_id, log_prob, old_value, action_embed, rew_embed in zip(
                observations, actions, rewards, batch_ids, log_probs, old_values, action_embeds, rew_embeds):
            if batch_id is None:
                continue
            self._observations_list[batch_id].append(observation.astype(np.float32))
            self._actions_list[batch_id].append(action.astype(np.float32))
            self._rewards_list[batch_id].append(reward.astype(np.float32))
            self._action_embed_list[batch_id].append(action_embed.astype(np.float32))
            self._rew_embed_list[batch_id].append(rew_embed.astype(np.float32))
            self._logprob_list[batch_id].append(log_prob.astype(np.float32))
            self._old_value_list[batch_id].append(old_value.astype(np.float32))
