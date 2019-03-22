import maml_rl.envs
import gym
import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from train_mdrnn import FFModel, one_hot

class Planner:
	def __init__(self, model, k, n):
		self.model = model
		self.k = k
		self.n = n

	def get_action(self, obs):
		best_traj = None; best_traj_rew = -1.0
		for n in range(self.n):
			actions = np.random.randint(0, self.model.action_dim, size=self.k)
			act_tens = torch.from_numpy(one_hot(actions, self.model.action_dim)).float()
			obs_tens = torch.from_numpy(obs.flatten()[None])
			rew = 0.

			with torch.no_grad():
				for i in range(len(actions)):
					obs_tens, r, d = model(obs_tens, act)
					rew += r.numpy()

			if rew >= best_traj_rew:
				best_traj = actions
		return actions[0]

def main(args):
	env = gym.make(args.env)
	obs_shape = env.observation_space.shape
	act_dim = env.action_space.n
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	model = FFModel(input_size=obs_shape[0] * obs_shape[1], action_dim=act_dim, hidden_size=256)
	model.load_state_dict(torch.load(args.model_file, 
    	map_location=device if device == 'cpu' else None))

	planner = Planner(model, k=args.k, n=args.n)
	tot_R = []; tot_T = []

	for t in range(args.test_eps):
		obs = env.reset() done = False
		ep_R = 0.; ep_T = 0
		while not done:
			action = planner.get_action(obs)
			obs, rew, done, info = env.step(action)
			ep_R += rew
			ep_T += 1
		tot_T.append(ep_T)
		tot_R.append(ep_R)

	print("MeanEpRew: ", np.mean(tot_R))
	print("MeanEpSteps: ", np.mean(tot_T))


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--env', type=str, default='GridGame-v0')
	parser.add_argument('--test_eps', type=int, default=10)
	parser.add_argument('--k', type=int, default=10, help='planning depth')
	parser.add_argument('--n', type=int, default=100, help='planning trajectories')
	parser.add_argument('--model_file', type=str)
	return parser.parse_args()

if __name__=="__main__":
	args = parse_args()
	main(args)
