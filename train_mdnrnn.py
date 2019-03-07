import maml_rl.envs
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

import time
import json
import argparse
import maml_rl.envs
import os
from tqdm import tqdm
from train_vae import BetaVAE
from torch.distributions.normal import Normal

def collect_batch_episodes(env, vae, test_eps=100):
	episodes = []
	for _ in range(test_eps):
		obs = env.reset(); done = False
		ep_obs = []; ep_dict = {}; ep_rew = []; ep_done = []; ep_act = []
		while not done:
			act = env.action_space.sample()
			obs, rew, done, info = env.step(act)
			ep_obs.append(obs); ep_rew.append(rew); ep_done.append(float(done)); ep_act.append(act)
		ep_obs = np.array(ep_obs, copy=False)
		ep_rew = np.array(ep_rew, copy=False)
		ep_done = np.array(ep_done, copy=False)
		ep_act = np.array(ep_act, copy=False)

		with torch.no_grad():
			mu, sig = vae.encode(torch.from_numpy(ep_obs))
			latent = vae.sample_latent(mu, sig)
		ep_dict['obs'] = latent
		ep_dict['done'] = torch.from_numpy(ep_done)
		ep_dict['rew'] = torch.from_numpy(ep_rew)
		ep_dict['act'] = torch.from_numpy(ep_act)
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
		self.gmm_linear = nn.Linear(self.lstm_size, self.input_size + 2)

	def forward(self, x, a, hx):
		out = torch.cat((x, a), dim=1)
		h_out = self.gru(out, hx)
		out = self.gmm_linear(h_out)
		preds = out[:, :-2]; rs = out[:, -2]; ds = out[:, -1]
		return preds, rs, ds, h_out

def main(args):
	env = gym.make('CustomGame-v0')
	obs_shape = env.observation_space.shape
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	vae = BetaVAE(input_size=obs_shape[-1], hidden_size=args.hidden).to(device)
	if os.path.exists(args.vae_path):
		vae.load_state_dict(torch.load(args.vae_path))
	vae.eval()

	mdn = MDNRNN(input_size=args.hidden, lstm_size=256, K=args.K)
	data = collect_batch_episodes(env, test_eps=args.T) # (T, L, hidden)
	N = len(data) # number of episodes
	optimizer = torch.optim.Adam(mdn.parameters(), lr=args.lr)

	# creating output dir
	if not os.path.exists(args.outdir):
		os.makedirs(args.outdir + '/', exist_ok=True)

	for ep in range(args.epochs):
		# Training
		mdn.train(); train_loss = []
		for batch_idx in range(N):
			data_dict = data[np.random.randint(N)].to(device)
			data_batch, rew_batch, done_batch, act_batch = data_dict['obs'], data_dict['rew'], data_dict['done'], data_dict['act']
			hx = torch.zeros(1, 256)
			for t in range(data_batch.size(0) - 1):
				optimizer.zero_grad()
				pred, r, d, hx = model(data_batch[t], act_batch[t], hx)

				# computing losses
				pred_loss = F.mse_loss(pred, data_batch[t+1])
				done_loss = F.binary_cross_entropy_with_logits(d, done_batch[t])
				rew_loss = F.mse_loss(r, rew_batch[t])
				loss = (pred_loss + done_loss + rew_loss) / (args.hidden + 2)
				loss.backward()
				train_loss.append(loss.item())
				optimizer.step()
		print('====> Epoch: {} TrainLoss: {:.4f}'.format(ep, np.mean(train_loss)))

	with open(os.path.join(args.outdir, 'final.pt'), 'wb') as f:
		torch.save(mdn.state_dict(), f)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--hidden', type=int, default=64, help='hidden size')
	parser.add_argument('--vae_path', type=str, help='path for vae model')
	parser.add_argument('--K', type=int, default=5, help='number of gaussians in GMM')
	parser.add_argument('--T', type=int, default=100, help='number of rollouts to collect')
	parser.add_argument('--outdir', type=str, default='results/', help='where to save results')
	parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
	parser.add_argument('--epochs', type=int, default=100, help='nmber train epochs')
	args = parser.parse_args()
	main(args)
