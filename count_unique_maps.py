import gym
import numpy as np
from tqdm import tqdm
import argparse
import maml_rl.envs

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='CustomGame-v0')
parser.add_argument('--test-eps', type=int, default=10000)
args = parser.parse_args()

unique_maps = []
env = gym.make(args.env)
for t in tqdm(range(args.test_eps)):
	obs = env.reset()
	map_t = env.unwrapped.game_state.game.newGame.map
	map_t[map_t == 21] = 0; map_t[map_t == 22] = 0
	map_string = ''.join(map(str, list(map_t)))
	if map_string not in unique_maps:
		unique_maps.append(map_string)

print("UniqueMaps: ", len(unique_maps))