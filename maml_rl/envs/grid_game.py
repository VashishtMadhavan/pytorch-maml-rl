import numpy as np
import gym
from gym import spaces
from gym.utils import seeding, colorize
from ple import PLE
import os
import random
import sys

def clear_path(map_x):
    rr = np.random.randint(1, 10, size=3)
    map_x[2, rr[0]] = 0.; map_x[2, rr[0] - 1] = 0.
    map_x[5, rr[1]] = 0.; map_x[5, rr[1] - 1] = 0.
    map_x[8, rr[2]] = 0.; map_x[8, rr[2] - 1] = 0.

def place_agents(map_x):
    pos = np.array([y for y in np.argwhere(map_x == 0)])
    choices = np.random.randint(pos.shape[0], size=2)
    x = pos[choices[0]]; y = pos[choices[1]]
    map_x[x[0], x[1]] = 2.
    map_x[y[0], y[1]] = 3.
    return x, y

def preprocess_map(map_x):
    map_x[map_x == 12] = 1 # removing init fire position
    map_x[map_x == 21] = 0 # removing init agent position
    map_x[map_x == 20] = 0 # removing init princess position
    map_x[map_x == 11] = 0 # removing init enemy position
    clear_path(map_x)
    return place_agents(map_x)

class GridGameEnv(gym.Env):
    def __init__(self, task={}):
        self._task = task
        map_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'map_base.txt')
        self.map = np.loadtxt(map_file, dtype='i', delimiter=',')
        self.agent_pos, self.goal_pos = preprocess_map(self.map)
        self.N, self.M = self.map.shape

        self.action_space = spaces.Discrete(5) #nothing up down left right
        self.observation_space = spaces.Box(low=0, high=255, shape=(len(self.map.flatten()), ))
        self.num_actions = 5
        self.reward_mult = 10.0
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
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")


    def reset(self):
        map_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'map_base.txt')
        self.map = np.loadtxt(map_file, dtype='i', delimiter=',')
        self.agent_pos, self.goal_pos = preprocess_map(self.map)
        return self.map.flatten() / 3.0

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
        self.map[self.agent_pos[0], self.agent_pos[1]] = 2.
        state = self.map.flatten() / 3.0
        done = list(self.agent_pos) == list(self.goal_pos)
        rew = float(done) * self.reward_mult
        return state, rew, done, {}