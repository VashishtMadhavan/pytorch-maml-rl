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

"""
Training beta-VAEs to learn disentagled representations
"""
class BetaVAE(nn.Module):
	def __init__(self, input_size, hidden_size=32):
		super(BetaVAE, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size

		self.encoder = nn.Sequential(
			nn.Conv2d(input_size, 32, kernel_size=4, stride=2),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 64, kernel_size=4, stride=2),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 128, kernel_size=4, stride=2),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 256, kernel_size=4, stride=2),
			nn.ReLU(inplace=True),
		)

		self.mu_fc = nn.Linear(2304, self.hidden_size)
		self.sigma_fc = nn.Linear(2304, self.hidden_size)
		self.decode_fc = nn.Linear(self.hidden_size, 1024)

		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(1024, 128, kernel_size=6, stride=2),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(128, 64, kernel_size=7, stride=2),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(64, 32, kernel_size=7, stride=2),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(32, input_size, kernel_size=8, stride=2),
			nn.Sigmoid(),
		)

	def encode(self, x):
		h = self.encoder(x)
		h = h.view(h.size(0), -1)
		return self.mu_fc(h), self.sigma_fc(h)

	def decode(self, z):
		h = self.decode_fc(z)
		h = h.unsqueeze(-1).unsqueeze(-1)
		return self.decoder(h)

	def sample_latent(self, mu, sigma):
		std = torch.exp(0.5 * sigma)
		eps = torch.randn_like(std)
		return eps.mul(std).add_(mu)

	def forward(self, x):
		mu, sigma = self.encode(x)
		z = self.sample_latent(mu, sigma)
		return self.decode(z), mu, sigma

class ConvAE(nn.Module):
	def __init__(self, input_size, hidden_size=32):
		super(ConvAE, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size

		self.encoder = nn.Sequential(
			nn.Conv2d(input_size, 32, kernel_size=4, stride=2),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 64, kernel_size=4, stride=2),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 128, kernel_size=4, stride=2),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 256, kernel_size=4, stride=2),
			nn.ReLU(inplace=True),
		)

		self.encode_fc = nn.Linear(2304, self.hidden_size)
		self.decode_fc = nn.Linear(self.hidden_size, 1024)

		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(1024, 128, kernel_size=6, stride=2),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(128, 64, kernel_size=7, stride=2),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(64, 32, kernel_size=7, stride=2),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(32, input_size, kernel_size=8, stride=2),
			nn.Sigmoid(),
		)

	def encode(self, x):
		h = self.encoder(x)
		h = h.view(h.size(0), -1)
		return self.encode_fc(h)

	def decode(self, z):
		h = self.decode_fc(z)
		h = h.unsqueeze(-1).unsqueeze(-1)
		return self.decoder(h)

	def forward(self, x):
		z = self.encode(x)
		return self.decode(z)

def vae_loss(x_pred, x, mu, sigma, beta=1.0):
	bce = F.binary_cross_entropy(x_pred, x, size_average=False)
	kl_div = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
	return bce + beta * kl_div

def vae_proj_loss(model, x_pred, x, mu, sigma, beta=1.0):
	z_pred = model.encode(x_pred)
	z = model.encode(x)
	bce = F.mse_loss(z_pred, z, size_average=False)
	kl_div = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
	return bce + beta * kl_div

def ae_loss(x_pred, x):
	bce = F.binary_cross_entropy(x_pred, x, size_average=False)
	return bce

def collect_random_episodes(env, eps=10):
	ep_obs = []
	for t in range(eps):
		done = False; obs = env.reset()
		while not done:
			obs, rew, done, info = env.step(env.action_space.sample())
			ep_obs.append(obs)
	return np.array(ep_obs)

def main(args):
	env = gym.make('CustomGame-v0')
	obs_shape = env.observation_space.shape
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model = BetaVAE(input_size=obs_shape[-1], hidden_size=args.hidden).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	proj_model = ConvAE(input_size=obs_shape[-1], hidden_size=args.hidden).to(device)
	proj_optim = torch.optim.Adam(proj_model.parameters(), lr=args.lr, weight_decay=1e-4)

	# Dataset Loading
	observations = collect_random_episodes(env, eps=100)
	N = observations.shape[0]
	np.random.shuffle(observations)
	trainX, testX = observations[:int(0.8 * N)], observations[int(0.8 * N):]

	# Pretrain Autencoder
	if not os.path.exists('pretrained.pt'):
		for p_ep in range(50):
			p_train_batches = len(trainX) // args.batch_size
			for idx in tqdm(range(p_train_batches)):
				data = trainX[np.random.choice(np.arange(len(trainX)), size=args.batch_size, replace=False)]
				data = torch.from_numpy(data).permute(0, 3, 1, 2).to(device)
				proj_optim.zero_grad()
				pred = proj_model(data)
				loss = ae_loss(pred, data)
				loss.backward()
				proj_optim.step()
		with open('pretrained.pt', 'wb') as f:
			torch.save(proj_model.state_dict(), f)
	else:
		proj_model.load_state_dict(torch.load('pretrained.pt'))
	for pp in proj_model.parameters():
		pp.requires_grad = False

	# checking if results dir exists
	if not os.path.exists(args.outdir + '/'):
		os.makedirs(args.outdir + '/', exist_ok=True)

	for ep in range(args.epochs):
		# Training
		model.train(); train_loss = []
		num_train_batches = len(trainX) // args.batch_size
		for batch_idx in range(num_train_batches):
			data = trainX[np.random.choice(np.arange(len(trainX)), size=args.batch_size, replace=False)]
			data = torch.from_numpy(data).permute(0, 3, 1, 2).to(device)
			optimizer.zero_grad()
			pred, mu, sigma = model(data)
			if args.enc:
				loss = vae_proj_loss(proj_model, pred, data, mu, sigma, beta=args.beta)
			else:
				loss = vae_loss(pred, data, mu, sigma, beta=args.beta)
			loss.backward()
			train_loss.append(loss.item())
			optimizer.step()

		# Testing
		model.eval(); test_loss = []
		num_test_batches = len(testX) // args.batch_size
		with torch.no_grad():
			for batch_idx in range(num_test_batches):
				data = testX[np.random.choice(np.arange(len(testX)), size=args.batch_size, replace=False)]
				data = torch.from_numpy(data).permute(0, 3, 1, 2).to(device)
				pred, mu, sigma = model(data)
				if args.enc:
					tloss = vae_proj_loss(proj_model, pred, data, mu, sigma, beta=args.beta)
				else:
					tloss = vae_loss(pred, data, mu, sigma, beta=args.beta)
				test_loss.append(tloss.item())
				if batch_idx == 0:
					n = 8
					comparison = torch.cat([data[:n], 
						pred.view(args.batch_size, obs_shape[2], obs_shape[0], obs_shape[1])[:n]])
					comparison = comparison[:, :1, :, :]
					save_image(comparison.cpu(), '{0}/reconstruction_{1}.png'.format(args.outdir, ep), nrow=n)

		print('====> Epoch: {} TrainLoss: {:.4f}  TestLoss: {:.4f}'.format(ep, np.mean(train_loss), np.mean(test_loss)))

		# Decoding random samples
		with torch.no_grad():
			z_sample = torch.randn(32, args.hidden).to(device)
			pred_sample = model.decode(z_sample).cpu()
			save_image(pred_sample.view(32, obs_shape[2], obs_shape[0], obs_shape[1])[:, :1, :, :], '{0}/sample_{1}.png'.format(args.outdir, ep))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--enc', action='store_true', help='to use random projections or not')
	parser.add_argument('--hidden', type=int, default=32, help='hidden size')
	parser.add_argument('--outdir', type=str, default='results/', help='where to save results')
	parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
	parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
	parser.add_argument('--epochs', type=int, default=100, help='nmber train epochs')
	parser.add_argument('--beta', type=float, default=1.0, help='beta for disentagled representations')
	args = parser.parse_args()
	main(args)
