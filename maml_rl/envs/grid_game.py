import numpy as np
import gym
from gym import spaces
from gym.utils import seeding, colorize
from ple import PLE
import os
import random
import sys

def place_agents(map_x, setup):
    post = []
    for j in [1.0, 2.0]:
        pos = np.array([y for y in np.argwhere(map_x == 0)])
        if setup == 0:
            pos = pos[pos[:,0] != 1]
        elif setup == 1:
            pos = pos[pos[:, 0] == 1]
        else:
            pass
        choice = np.random.randint(pos.shape[0])
        x = pos[choice]
        map_x[x[0]][x[1]] = j
        post.append(x)
    return post[0], post[1]

def preprocess_map(map_x, setup):
    return place_agents(map_x, setup)

class GridGameEnv(gym.Env):
    def __init__(self, task={}, setup=0, dim=5):
        self._task = task
        self.setup = setup
        self.map_dim = dim
        self.map = np.zeros((self.map_dim, self.map_dim))
        self.agent_pos, self.goal_pos = preprocess_map(self.map, self.setup)
        self.init_apos = np.array(self.agent_pos)
        self.init_gpos = np.array(self.goal_pos)
        self.N, self.M = self.map.shape

        self.action_space = spaces.Discrete(5) #nothing up down left right
        self.observation_space = spaces.Box(low=0, high=255, shape=self.map.shape)
        self.num_actions = 5
        self.viewer = None

    def seed(self, seed=None):
        if not seed:
            seed = np.random.randint(2**31-1)
        rng = np.random.RandomState(seed)
        return [seed]

    def reset_task(self, task):
        pass

    def render(self, mode='human'):
        outfile = sys.stdout
        desc = self.map.tolist()
        desc = [[str(c) for c in line] for line in desc]
        desc[self.agent_pos[0]][self.agent_pos[1]] = colorize(desc[self.agent_pos[0]][self.agent_pos[1]], "red", highlight=True)
        desc[self.goal_pos[0]][self.goal_pos[1]] = colorize(desc[self.goal_pos[0]][self.goal_pos[1]], "blue", highlight=True)
        outfile.write("\n")
        outfile.write("\n".join(' '.join(line) for line in desc)+"\n")


    def reset(self):
        self.map = np.zeros((self.map_dim, self.map_dim))
        self.agent_pos, self.goal_pos = preprocess_map(self.map, self.setup)
        return self.map.astype(np.float32) / 2.0

    def step(self, action):
        self.map[self.agent_pos[0], self.agent_pos[1]] = 0.
        pos = np.array(self.agent_pos)
        if action == 0:
            pass
        elif action == 1:
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 2:
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.N - 1)
        elif action == 3:
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif action == 4:
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.M - 1)
        if self.map[self.agent_pos[0], self.agent_pos[1]] == 1:
            self.agent_pos = pos
        self.map[self.agent_pos[0], self.agent_pos[1]] = 1.0
        state = self.map.astype(np.float32) / 2.0
        done = list(self.agent_pos) == list(self.goal_pos)
        rew = float(done)
        return state, rew, done, {}