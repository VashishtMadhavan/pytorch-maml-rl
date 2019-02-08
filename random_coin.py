import numpy as np
from tqdm import tqdm
import argparse
import maml_rl.envs


parser = argparse.ArgumentParser()
parser.add_argument('--test-eps', type=int, default=500)
parser.add_argument('--render', action='store_true')
args = parser.parse_args()

total_rew = []
total_steps = []

env = maml_rl.envs.CoinRunEnv(game_type='standard', num_envs=4, frame_stack=1)
obs = env.reset(); done = False
rews = np.zeros(env.num_envs)
done_count = 0; step_count = 0

while done_count < args.test_eps:
	if args.render:
		env.render()
	action = np.array([env.action_space.sample() for _ in range(env.num_envs)])
	obs, rew, done, info = env.step(action)
	rews += rew
	done_count += np.sum(done)
	step_count += 1

print("MeanReward: ", np.sum(rews) / args.test_eps)
print("MeanSteps: ", step_count * 4)