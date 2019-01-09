import gym
import numpy as np
from tqdm import tqdm
import argparse
import maml_rl.envs


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='CustomGame-v0')
parser.add_argument('--test-eps', type=int, default=500)
parser.add_argument('--render', action='store_true')
args = parser.parse_args()

total_rew = []
total_steps = []

env = gym.make(args.env)
for t in tqdm(range(args.test_eps)):
	obs = env.reset(); done = False
	steps = 0; ep_rew = 0
	while not done:
		if args.render:
			env.render()
		obs, rew, done, info = env.step(np.random.randint(env.action_space.n))
		ep_rew += rew
		steps += 1
	total_steps.append(steps)
	total_rew.append(ep_rew)

print("EpisodeRewMean: ", np.mean(total_rew))
print("EpisodeStepsMean: ", np.mean(total_steps))



