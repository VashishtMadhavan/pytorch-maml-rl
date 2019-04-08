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
		self.fc = nn.Linear(4 * 4 * 16, 32)
		self.final = nn.Linear(32, self.output_size)

	def forward(self, x):
		out = x.unsqueeze(1)
		out = F.relu(self.conv1(out))
		out = F.relu(self.conv2(out))
		out = F.relu(self.conv3(out))
		out = out.view(out.size(0), -1)
		out = F.relu(self.fc(out))
		return self.final(out)

def get_batch(data, batch_size, device=torch.device('cpu')):
	random_idx = np.random.choice(np.arange(len(data)), size=batch_size, replace=False)
	data_dicts = [data[r] for r in random_idx]

	obs_batch = torch.cat([d['obs'] for d in data_dicts], dim=0)
	dir_batch = torch.cat([d['dir'] for d in data_dicts], dim=0)
	return obs_batch.to(device), dir_batch.to(device)

def main(args):
	env = gym.make('GridGameTrain-v0')
	test_env = gym.make('GridGameSmallTest-v0')
	full_test_env = gym.make('GridGameTest-v0')
	obs_shape = env.observation_space.shape
	act_dim = env.action_space.n
	if not os.path.exists(args.outdir):
		os.makedirs(args.outdir + '/', exist_ok=True)

	model = DirectionModel(input_size=obs_shape[0] * obs_shape[1], output_size=NUM_DIRECTIONS)
	model.to(args.device)

	# Training
	data = collect_batch_episodes(env, test_eps=args.T, conv=True) # (T, L, hidden)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_pen)

	random.shuffle(data)
	train_data, test_data = data[:int(0.8 * args.T)], data[int(0.8 * args.T):]
	for ep in range(args.epochs):
		# Training
		model.train()
		num_batches = len(train_data) // args.batch_size
		avg_train_loss = []; train_correct = 0; train_count = 0
		for idx in tqdm(range(num_batches)):
			obs_batch, dir_batch  = get_batch(train_data, args.batch_size, device=args.device)
			optimizer.zero_grad()
			pred = model(obs_batch)
			loss = F.cross_entropy(pred, dir_batch)
			loss.backward()
			pred_class = F.log_softmax(pred, dim=1).argmax(dim=1)
			train_correct += pred_class.eq(dir_batch).sum().item()
			train_count += pred_class.shape[0]
			avg_train_loss.append(loss.item())
			optimizer.step()
		avg_train_acc = float(train_correct) / train_count

		model.eval()
		t_obs_batch, t_dir_batch = get_batch(test_data, 1000, device=args.device)
		with torch.no_grad():
			t_pred = model(t_obs_batch)
			t_loss = F.cross_entropy(t_pred, t_dir_batch)
			t_pred_class = F.log_softmax(t_pred, dim=1).argmax(dim=1)
			t_pred_acc = t_pred_class.eq(t_dir_batch).sum().item() / t_pred_class.shape[0]
		print('====> Epoch: {} TrainLoss: {:.4f} TrainAcc: {:.4f} TestLoss: {:.4f} TestAcc: {:.4f}'.format(ep, np.mean(avg_train_loss), avg_train_acc, t_loss.item(), t_pred_acc))

	with open(os.path.join(args.outdir, 'final.pt'), 'wb') as f:
		torch.save(model.state_dict(), f)

	# Testing
	test_data = collect_batch_episodes(test_env, test_eps=1000, conv=True)
	full_test_data = collect_batch_episodes(full_test_env, test_eps=1000, conv=True)

	num_test_batches = len(test_data) // 100
	num_full_test_batches = len(full_test_data) // 100
	with torch.no_grad():
		test_correct = 0; test_count = 0
		for tidx in range(num_test_batches):
			test_obs_batch, test_dir_batch = get_batch(test_data, 100, device=args.device)
			test_pred_class = F.log_softmax(model(test_obs_batch), dim=1).argmax(dim=1)
			test_correct += test_pred_class.eq(test_dir_batch).sum().item()
			test_count += test_pred_class.shape[0]
		test_acc = float(test_correct) / test_count

		test_correct = 0; test_count = 0
		for tidx in range(num_test_batches):
			test_obs_batch, test_dir_batch = get_batch(full_test_data, 100, device=args.device)
			test_pred_class = F.log_softmax(model(test_obs_batch), dim=1).argmax(dim=1)
			test_correct += test_pred_class.eq(test_dir_batch).sum().item()
			test_count += test_pred_class.shape[0]
		full_test_acc = float(test_correct) / test_count
	print("SmallTestAcc: ", test_acc)
	print("FullTestAcc: ", full_test_acc)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
	parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
	parser.add_argument('--T', type=int, default=20000, help='number of rollouts to collect')
	parser.add_argument('--outdir', type=str, default='dir_model_debug/', help='where to save results')
	parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
	parser.add_argument('--l2_pen', type=float, default=1e-3, help='l2 regularization penalty')
	parser.add_argument('--epochs', type=int, default=25, help='number of training epochs')
	args = parser.parse_args()

	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
	args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	main(args)
