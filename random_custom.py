import gym
import numpy as np
from tqdm import tqdm
import maml_rl.envs

ENV_NAME = 'CustomGame-v0'
TEST_EPS = 1000
RENDER = False

total_rew = []
total_steps = []

env = gym.make(ENV_NAME)
for t in tqdm(range(TEST_EPS)):
	obs = env.reset(); done = False
	steps = 0; ep_rew = 0
	while not done:
		if RENDER:
			env.render()
		obs, rew, done, info = env.step(np.random.randint(env.action_space.n))
		ep_rew += rew
		steps += 1
	total_steps.append(steps)
	total_rew.append(ep_rew)

print("EpisodeRewMean: ", np.mean(total_rew))
print("EpisodeStepsMean: ", np.mean(total_steps))



