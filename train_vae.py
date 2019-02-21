import maml_rl.envs
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import time
import json
import argparse
import maml_rl.envs
import os

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

		# TODO: make sure this hidden dim is correct
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

def vae_loss(x_pred, x, mu, sigma, beta=1.0):
	bce = F.binary_cross_entropy(x_pred, x, size_average=False)
	kl_div = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
	return bce + beta * kl_div

def main(args):
	env = gym.make('CustomGame-v0')
	obs_shape = env.observation_space.shape
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	data_kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}

	model = BetaVAE(input_size=1, hidden_size=args.hidden).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	# Dataset Loading
	# TOOD: replace with GYM data that was collected....
	data_transform = transforms.Compose([
		transforms.Resize((84, 84)), 
		transforms.ToTensor()])

	train_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist', train=True, download=True,
		transform=data_transform), batch_size=args.batch_size, shuffle=True, **data_kwargs)

	test_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist', train=False, 
		transform=data_transform), batch_size=args.batch_size, shuffle=True, **data_kwargs)

	# checking if results dir exists
	if not os.path.exists(args.outdir + '/'):
		os.makedirs(args.outdir + '/', exist_ok=True)

	for ep in range(args.epochs):
		# Training
		model.train(); train_loss = []
		for batch_idx, (data, _) in enumerate(train_loader):
			data = data.to(device)
			optimizer.zero_grad()
			pred, mu, sigma = model(data)
			loss = vae_loss(pred, data, mu, sigma, beta=args.beta)
			loss.backward()
			train_loss.append(loss.item())
			optimizer.step()

		# Testing
		model.eval(); test_loss = []
		with torch.no_grad():
			for batch_idx, (data, _) in enumerate(test_loader):
				data = data.to(device)
				pred, mu, sigma = model(data)
				tloss = vae_loss(pred, data, mu, sigma, beta=args.beta)
				test_loss.append(tloss.item())
				if batch_idx == 0:
					n = 8
					comparison = torch.cat([data[:n], 
						pred.view(args.batch_size, 1, 84, 84)[:n]])
					save_image(comparison.cpu(), '{0}/reconstruction_{1}.png'.format(args.outdir, ep), nrow=n)

		print('====> Epoch: {} TrainLoss: {:.4f}  TestLoss: {:.4f}'.format(ep, np.mean(train_loss), np.mean(test_loss)))

		# Decoding random samples
		with torch.no_grad():
			z_sample = torch.randn(32, args.hidden).to(device)
			pred_sample = model.decode(z_sample).cpu()
			save_image(pred_sample.view(32, 1, 84, 84), '{0}/sample_{1}.png'.format(args.outdir, ep))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--hidden', type=int, default=32, help='hidden size')
	parser.add_argument('--outdir', type=str, default='results/', help='where to save results')
	parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
	parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
	parser.add_argument('--epochs', type=int, default=10, help='nmber train epochs')
	parser.add_argument('--beta', type=float, default=1.0, help='beta for disentagled representations')
	args = parser.parse_args()
	main(args)
