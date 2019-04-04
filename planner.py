import maml_rl.envs
import gym
import argparse
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from train_mdnrnn import FFModel, one_hot

class Planner:
	def __init__(self, model, k, n):
		self.model = model
		self.k = k
		self.n = n

	def get_action(self, obs, device=torch.device('cpu')):
		obs_input = np.repeat(obs.flatten()[None], self.n, axis=0) # [N, obs_dim]
		actions = np.random.randint(0, self.model.action_dim, size=(self.n, self.k))
		actions_one_hot = np.array([one_hot(a, self.model.action_dim) for a in actions])

		act_tens = torch.from_numpy(actions_one_hot).float().to(device)
		obs_tens = torch.from_numpy(obs_input).to(device)
		rew = np.zeros(self.n)

		with torch.no_grad():
			for t in range(actions.shape[1]):
				obs_tens, r = self.model(obs_tens, act_tens[:, t, :])
				rew += np.round(F.sigmoid(r).numpy().squeeze())
		best_idx = np.argmax(rew)
		return actions[best_idx][0]

def main(args):
	env = gym.make(args.env)
	obs_shape = env.observation_space.shape
	act_dim = env.action_space.n
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	model = FFModel(input_size=obs_shape[0] * obs_shape[1], action_dim=act_dim, hidden_size=256)
	model.to(device)
	model.load_state_dict(torch.load(args.model_file, 
    	map_location=device if device == 'cpu' else None))

	planner = Planner(model, k=args.k, n=args.n)
	tot_R = []; tot_T = []

	for t in tqdm(range(args.test_eps)):
		obs = env.reset(); done = False
		ep_R = 0.; ep_T = 0
		while not done:
			action = planner.get_action(obs, device=args.device)
			obs, rew, done, info = env.step(action)
			ep_R += rew
			ep_T += 1
		tot_T.append(ep_T)
		tot_R.append(ep_R)

	print("MeanEpRew: ", np.mean(tot_R))
	print("StdMeanRew: ", np.std(tot_R) / np.sqrt(len(tot_R)))
	print("MeanEpSteps: ", np.mean(tot_T))


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--env', type=str, default='GridGameTrain-v0')
	parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
	parser.add_argument('--test_eps', type=int, default=1000)
	parser.add_argument('--k', type=int, default=5, help='planning depth')
	parser.add_argument('--n', type=int, default=500, help='planning trajectories')
	parser.add_argument('--model_file', type=str)
	return parser.parse_args()

	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__=="__main__":
	args = parse_args()
	main(args)
