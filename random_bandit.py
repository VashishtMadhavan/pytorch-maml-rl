import gym
import numpy as np
from tqdm import tqdm
import argparse
import maml_rl.envs


parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--n', type=int, default=10)
parser.add_argument('--test-eps', type=int, default=500)
args = parser.parse_args()

env_name = 'Bandit-K{0}-N{1}-v0'.format(args.k, args.n)
epR = []; epT = []
env = gym.make(env_name)

for t in tqdm(range(args.test_eps)):
	obs = env.reset(); done = False
	T = 0; R = 0
	while not done:
		obs, rew, done, info = env.step(np.random.randint(env.action_space.n))
		R += rew; T += 1
	epT.append(T); epR.append(R)

print("EpisodeRewMean: ", np.mean(epR))
print("EpisodeStepsMean: ", np.mean(epT))



