import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from ple import PLE
import os
import random

class CustomGameEnv(gym.Env):
    def __init__(self, task={}):
        self._task = task
        os.environ['SDL_VIDEODRIVER'] = 'dummy'

        import importlib
        game_module = importlib.import_module('ple.games.customgame')
        game = getattr(game_module, 'customgame')(difficulty=0)

        self.game_state = PLE(game, fps=30, display_screen=False)
        self._action_set = self.game_state.getActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))
        self.screen_width, self.screen_height = self.game_state.getScreenDims()
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height, 3))

        self.num_actions = len(self._action_set)
        self.viewer = None

    def seed(self, seed=None):
        if not seed:
            seed = np.random.randint(2**31-1)
        rng = np.random.RandomState(seed)
        self.game_state.rng = rng
        self.game_state.game.rng = self.game_state.rng
        self.game_state.init()
        return [seed]

    # TODO: figure this out
    def reset_task(self, task):
        self.game_state.game.easy_env_flag = 1.0 - self.game_state.game.easy_env_flag

    def render(self, mode='human'):
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

    def reset(self):
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height, 3))
        self.game_state.reset_game()
        state = self._get_image()
        return state
        
    def _get_image(self):
        image_rotated = np.fliplr(np.rot90(self.game_state.getScreenRGB(),3)) # Hack to fix the rotated image returned by ple
        return image_rotated

    def step(self, action):
        reward = self.game_state.act(self._action_set[action])
        state = self._get_image()
        terminal = self.game_state.game_over()
        return state, reward, terminal, {}
