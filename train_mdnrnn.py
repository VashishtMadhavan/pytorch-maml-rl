import maml_rl.envs
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

import time
import random
import json
import argparse
import maml_rl.envs
import os
from tqdm import tqdm
from train_vae import BetaVAE
from torch.distributions.normal import Normal

def one_hot(action, num_actions):
	x = np.zeros((len(action), num_actions))
	x[np.arange(len(action)), action] = 1.
	return x

def collect_batch_episodes(env, test_eps=100):
	episodes = []
	for _ in tqdm(range(test_eps)):
		obs = env.reset(); done = False
		ep_obs = []; ep_dict = {}; ep_rew = []; ep_done = []; ep_act = []
		ep_obs.append(obs)
		while not done:
			act = env.action_space.sample()
			obs, rew, done, info = env.step(act)
			ep_obs.append(obs); ep_rew.append(rew); ep_done.append(float(done)); ep_act.append(act)
		ep_obs = np.array(ep_obs, copy=False)
		ep_rew = np.array(ep_rew, copy=False)
		ep_done = np.array(ep_done, copy=False)
		ep_act = np.array(ep_act, copy=False)

		ep_dict['obs'] = torch.from_numpy(ep_obs)
		ep_dict['done'] = torch.from_numpy(ep_done).float()
		ep_dict['rew'] = torch.from_numpy(ep_rew).float()
		ep_dict['act'] = torch.from_numpy(one_hot(ep_act, env.action_space.n)).float()
		episodes.append(ep_dict)
	return episodes

class MDNRNN(nn.Module):
	def __init__(self, input_size, action_dim, lstm_size, K=2):
		super(MDNRNN, self).__init__()
		self.input_size = input_size
		self.action_dim = action_dim
		self.lstm_size = lstm_size
		self.K = K # number of gaussians in GMM

		self.gru = nn.GRUCell(self.input_size + self.action_dim, self.lstm_size)
		self.linear = nn.Linear(self.lstm_size, self.input_size + 2)

	def forward(self, x, a, hx):
		out = torch.cat((x, a), dim=-1)
		h_out = self.gru(out, hx)
		out = self.linear(h_out)
		preds = out[:, :-2]; rs = out[:, -2]; ds = out[:, -1]
		return preds, rs, ds, h_out

def get_batch(data, batch_size):
	random_idx = np.random.choice(np.arange(len(data)), size=batch_size, replace=False)
	data_dicts = [data[r] for r in random_idx]
	data_batch = [d['obs'] for d in data_dicts]
	act_batch = [d['act'] for d in data_dicts]
	rew_batch = [d['rew'] for d in data_dicts]
	done_batch = [d['done'] for d in data_dicts]
	return data_batch, act_batch, rew_batch, done_batch

def get_loss(data, act, rew, done, model):
	total_loss = 0; obs_shape = data[0][0].shape
	for bs in range(len(data)):
		hx = torch.zeros(1, model.lstm_size)
		for t in range(data[bs].size(0) - 1):
			pred, r, d, hx = model(data[bs][t].unsqueeze(0), act[bs][t].unsqueeze(0), hx)

			# computing losses
			pred_loss = F.mse_loss(pred, data[bs][t+1].unsqueeze(0))
			done_loss = F.binary_cross_entropy_with_logits(d, done[bs][t].unsqueeze(0))
			rew_loss = F.mse_loss(r, rew[bs][t].unsqueeze(0))
			total_loss += (pred_loss + done_loss + rew_loss) / (obs_shape[0] + 2)
	return total_loss / len(data)

def main(args):
	env = gym.make('GridGame-v0')
	obs_shape = env.observation_space.shape
	act_dim = env.action_space.n
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	mdn = MDNRNN(input_size=obs_shape[0], action_dim=act_dim, lstm_size=args.lstm_size, K=args.K)
	data = collect_batch_episodes(env, test_eps=args.T) # (T, L, hidden)
	optimizer = torch.optim.Adam(mdn.parameters(), lr=args.lr)

	random.shuffle(data)
	train_data, test_data = data[:int(0.8 * args.T)], data[int(0.8 * args.T):]

	# creating output dir
	if not os.path.exists(args.outdir):
		os.makedirs(args.outdir + '/', exist_ok=True)

	for ep in range(args.epochs):
		# Training
		mdn.train()
		num_batches = len(train_data) // args.batch_size
		for idx in tqdm(range(num_batches)):
			data_batch, act_batch, rew_batch, done_batch = get_batch(train_data, args.batch_size)
			optimizer.zero_grad()
			ave_loss = get_loss(data_batch, act_batch, rew_batch, done_batch, mdn)
			ave_loss.backward()
			optimizer.step()

		mdn.eval()
		test_data_batch, test_act_batch, test_rew_batch, test_done_batch = get_batch(test_data, 100)
		with torch.no_grad():
			avg_t_loss = get_loss(test_data_batch, test_act_batch, test_rew_batch, test_done_batch, mdn)
		print('====> Epoch: {} TestLoss: {:.4f}'.format(ep, avg_t_loss.item()))

	with open(os.path.join(args.outdir, 'final.pt'), 'wb') as f:
		torch.save(mdn.state_dict(), f)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--lstm_size', type=int, default=32, help='lstm hidden size')
	parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
	parser.add_argument('--K', type=int, default=5, help='number of gaussians in GMM')
	parser.add_argument('--T', type=int, default=10000, help='number of rollouts to collect')
	parser.add_argument('--outdir', type=str, default='mdn_debug/', help='where to save results')
	parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
	parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
	args = parser.parse_args()
	main(args)
