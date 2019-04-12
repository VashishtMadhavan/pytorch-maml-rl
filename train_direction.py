import maml_rl.envs
import gym
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import random
import json
import argparse
import maml_rl.envs
import pickle
import os
from tqdm import tqdm

"""
Training a model that learns to predict the direction of the goal
"""

NUM_DIRECTIONS = 8
def compute_direction(obs):
	goal_x, goal_y = np.where(obs == 1)[0][0], np.where(obs == 1)[1][0]
	agent_x, agent_y = np.where(obs == 0.5)[0][0], np.where(obs == 0.5)[1][0]
	rad = math.atan2(goal_y - agent_y, goal_x - agent_x)
	deg = math.degrees(rad) + 180.0
	class_id = deg // 45
	if class_id == 8:
		return 0
	return int(class_id)

def collect_batch_episodes(env, test_eps=100, conv=False):
	episodes = []
	for _ in tqdm(range(test_eps)):
		obs = env.reset(); done = False
		ep_obs = []; ep_dict = {}; ep_dir = []
		while not done:
			ep_obs.append(obs); ep_dir.append(compute_direction(obs))
			act = np.random.randint(env.action_space.n)
			obs, rew, done, info = env.step(act)

		ep_obs = np.array(ep_obs, copy=False)
		ep_dir = np.array(ep_dir, copy=False)
		ep_dict['obs'] = torch.from_numpy(ep_obs)
		ep_dict['dir'] = torch.from_numpy(ep_dir)
		episodes.append(ep_dict)
	return episodes

class DirectionModel(nn.Module):
	def __init__(self, input_size, output_size):
		super(DirectionModel, self).__init__()
		self.input_size = input_size
		self.output_size = output_size

		self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
		self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
		self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
		self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
		self.fc = nn.Linear(2 * 2 * 16, 32)
		self.final = nn.Linear(32, self.output_size)

	def forward(self, x):
		out = x.unsqueeze(1)
		out = F.relu(self.conv1(out))
		out = F.relu(self.conv2(out))
		out = F.relu(self.conv3(out))
		out = F.relu(self.conv4(out))
		out = out.view(out.size(0), -1)
		out = F.relu(self.fc(out))
		return self.final(out)


class DirectionTrainer:
	def __init__(self, data, obs_shape, args):
		self.model = DirectionModel(input_size=obs_shape[0] * obs_shape[1], output_size=NUM_DIRECTIONS)
		self.model.to(args.device)
		self.args = args
		self.obs = torch.cat([d['obs'] for d in data], dim=0)
		self.dir = torch.cat([d['dir'] for d in data], dim=0)

		self.N = len(self.obs)
		self.T = self.args.epochs
		self.device = self.args.device
		self.train_idx = np.random.choice(np.arange(self.N), size=int(0.8 * self.N), replace=False)
		self.val_idx = np.delete(np.arange(self.N), self.train_idx)
		self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.l2_pen)


	def eval(self, data, labels):
		self.model.eval()
		N = data.shape[0]
		num_batches = N // 1000

		test_correct = 0; test_count = 0
		test_loss = 0.
		with torch.no_grad():
			for n in range(num_batches):
				batch_idx = np.random.choice(N, size=1000, replace=False)
				obs = data[batch_idx].to(self.device); _dir = labels[batch_idx].to(self.device)
				pred = self.model(obs)
				pred_c = F.log_softmax(pred, dim=1).argmax(dim=1)
				loss = F.cross_entropy(pred, _dir)
				p_correct = pred_c.eq(_dir).sum().item()
				p_count = pred_c.shape[0]

				test_correct += p_correct
				test_count += p_count
				test_loss += loss.item()
		return test_loss / test_count, test_correct / test_count

	def train(self):
		self.model.train()
		for t in range(self.T):
			num_batches = self.train_idx.shape[0] // self.args.batch_size
			train_loss = 0.; train_acc = 0; train_count = 0
			for _ in tqdm(range(num_batches)):
				self.optim.zero_grad()
				batch_idx = np.random.choice(self.train_idx, size=self.args.batch_size, replace=False)
				obs = self.obs[batch_idx].to(self.device); _dir = self.dir[batch_idx].to(self.device)
				pred = self.model(obs)
				pred_c = F.log_softmax(pred, dim=1).argmax(dim=1)
				loss = F.cross_entropy(pred, _dir)
				p_correct = pred_c.eq(_dir).sum().item()
				p_count = pred_c.shape[0]
				loss.backward()

				train_acc += p_correct
				train_count += p_count
				train_loss += loss.item()
				self.optim.step()
			train_acc /= float(train_count)
			train_loss /= train_count

			# validation
			test_loss, test_acc = self.eval(self.obs[self.val_idx], self.dir[self.val_idx])
			print('====> Epoch: {} TrainLoss: {:.4f} TrainAcc: {:.4f} TestLoss: {:.4f} TestAcc: {:.4f}'.format(t, train_loss, train_acc, test_loss, test_acc))

	def save(self):
		with open(os.path.join(self.args.outdir, 'final.pt'), 'wb') as f:
			torch.save(self.model.state_dict(), f)


def main(args):
	env = gym.make('GridGameTrain-v0')
	test_env = gym.make('GridGameSmallTest-v0')
	full_test_env = gym.make('GridGameTest-v0')
	obs_shape = env.observation_space.shape
	act_dim = env.action_space.n
	if not os.path.exists(args.outdir):
		os.makedirs(args.outdir + '/', exist_ok=True)

	# Gather Training + Testing Data
	train_data = collect_batch_episodes(env, test_eps=args.T, conv=True) # (T, L, hidden)

	# Training
	trainer = DirectionTrainer(train_data, obs_shape, args)
	trainer.train()
	trainer.save()

	test_data = collect_batch_episodes(test_env, test_eps=1000, conv=True)
	full_test_data = collect_batch_episodes(full_test_env, test_eps=1000, conv=True)

	# small testing
	small_test_obs = torch.cat([d['obs'] for d in test_data], dim=0)
	small_test_dir = torch.cat([d['dir'] for d in test_data], dim=0)
	small_test_loss, small_test_acc = trainer.eval(small_test_obs, small_test_dir)

	# testing
	test_obs = torch.cat([d['obs'] for d in full_test_data], dim=0)
	test_dir = torch.cat([d['dir'] for d in full_test_data], dim=0)
	test_loss, test_acc = trainer.eval(test_obs, test_dir)

	print("SmallTestAcc: ", small_test_acc)
	print("FullTestAcc: ", test_acc)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
	parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
	parser.add_argument('--T', type=int, default=5000, help='number of rollouts to collect')
	parser.add_argument('--outdir', type=str, default='dir_model_debug/', help='where to save results')
	parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
	parser.add_argument('--l2_pen', type=float, default=1e-4, help='l2 regularization penalty')
	parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
	args = parser.parse_args()

	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
	args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	main(args)
