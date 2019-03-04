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
		ep_obs = []; ep_dict = {}; ep_rew = []; ep_done = []
		while not done:
			obs, rew, done, info = env.step(env.action_space.sample())
			ep_obs.append(obs); ep_rew.append(rew); ep_done.append(float(done))
		ep_obs = np.array(ep_obs, copy=False)
		ep_rew = np.array(ep_rew, copy=False)
		ep_done = np.array(ep_done, copy=False)

		with torch.no_grad():
			mu, sig = vae.encode(torch.from_numpy(ep_obs))
			latent = vae.sample_latent(mu, sig)
		ep_dict['obs'] = latent
		ep_dict['done'] = torch.from_numpy(ep_done)
		ep_dict['rew'] = torch.from_numpy(ep_rew)
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
		self.gmm_linear = nn.Linear(self.lstm_size, (2 * self.input_size + 1) * self.K + 2)

	def forward(self, x, a, hx):
		out = torch.cat((x, a), dim=1)
		h_out = self.gru(out, hx)
		out = self.gmm_linear(h_out)

		stride = self.input_size * self.K
		mus = out[:, :stride]
		mus = mus.view(mus.size(0), self.K, self.input_size)

		sigmas = out[:, stride:2*stride]
		sigmas = sigmas.view(sigmas.size(0), self.K, self.input_size)
		sigmas = torch.exp(sigmas)

		pi = out[:, 2*stride:2*stride + self.K]
		pi = pi.view(pi.size(0), self.K)
		logpi = F.log_softmax(pi, dim=-1)
		rs = out[:, -2]; ds = out[:, -1]
		return mus, sigmas, pi, rs, ds, h_out

def gmm_loss(z_next, mus, sigmas, logpi, reduce=True):
	# each tensor is (BS, K, input_size)
	normal_dist = Normal(mus, sigmas)
	g_log_probs = normal_dist.log_prob(z_next)
	g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
	max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
	g_log_probs = g_log_probs - max_log_probs

	probs = torch.sum(torch.exp(g_log_probs), dim=-1)
	log_prob = max_log_probs.squeeze() + torch.log(probs)

	if reduce:
		return -torch.mean(log_prob)
	return -log_prob

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
			data_batch, rew_batch, done_batch = data_dict['obs'], data_dict['rew'], data_dict['done']
			hx = torch.zeros(1, 256)
			for t in range(data_batch.size(0) - 1):
				optimizer.zero_grad()
				mu, sigma, pi, r, d, hx = model(data_batch[t])

				# computing losses
				pred_loss = gmm_loss(data_batch[t+1], mu, sigma, logpi)
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
