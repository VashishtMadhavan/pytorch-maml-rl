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
import pickle
import os
from tqdm import tqdm
from train_vae import BetaVAE
from torch.distributions.normal import Normal

class LinearSchedule(object):
	def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
		self.schedule_timesteps = schedule_timesteps
		self.final_p = final_p
		self.initial_p = initial_p

	def value(self, t):
		fraction  = min(float(t) / self.schedule_timesteps, 1.0)
		return self.initial_p + fraction * (self.final_p - self.initial_p)

class ConstantSchedule(object):
	def __init__(self, schedule_timesteps, p=1.0):
		self.schedule_timesteps = schedule_timesteps
		self.p = p

	def value(self, t):
		return self.p

def one_hot(action, num_actions):
	x = np.zeros((len(action), num_actions))
	x[np.arange(len(action)), action] = 1.
	return x

def collect_batch_episodes(env, test_eps=100, conv=False):
	episodes = []
	for _ in tqdm(range(test_eps)):
		obs = env.reset(); done = False
		ep_obs = []; ep_dict = {}; ep_rew = []; ep_done = []; ep_act = []
		if not conv:
			ep_obs.append(obs.flatten())
		else:
			ep_obs.append(obs)
		while not done:
			act = np.random.randint(env.action_space.n)
			obs, rew, done, info = env.step(act)
			if conv:
				ep_obs.append(obs)
			else:
				ep_obs.append(obs.flatten())
			ep_rew.append(rew); ep_done.append(float(done)); ep_act.append(act)
		ep_obs = np.array(ep_obs, copy=False)
		ep_rew = np.array(ep_rew, copy=False)
		ep_done = np.array(ep_done, copy=False)
		ep_act = np.array(ep_act, copy=False)

		prev_obs = np.vstack([np.zeros(ep_obs[0].shape)[None], ep_obs[:-2]])
		prev_act = np.insert(ep_act[:-1], 0, 0)

		ep_dict['prev_obs'] = torch.from_numpy(prev_obs).float()
		ep_dict['prev_act'] = torch.from_numpy(one_hot(prev_act, env.action_space.n)).float()
		ep_dict['obs'] = torch.from_numpy(ep_obs[:-1])
		ep_dict['next_obs'] = torch.from_numpy(ep_obs[1:])
		ep_dict['done'] = torch.from_numpy(ep_done).float()
		ep_dict['rew'] = torch.from_numpy(ep_rew).float()
		ep_dict['act'] = torch.from_numpy(one_hot(ep_act, env.action_space.n)).float()
		episodes.append(ep_dict)
	return episodes

class Model(nn.Module):
	def __init__(self, input_size, action_dim, conv=False, K=2):
		super(Model, self).__init__()
		self.input_size = input_size
		self.input_h = int(np.sqrt(input_size))
		self.action_dim = action_dim
		self.K = K
		self.conv = conv

		if self.conv:
			self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
			self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
			self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
			self.fc = nn.Linear(4 * 4 * 16 + self.action_dim, 32)
		else:
			self.fc = nn.Linear(self.input_size + self.action_dim, 32)
		self.rew_out = nn.Linear(32, 1)
		self.pred_out = nn.Linear(32, self.input_size)

	def forward(self, x, a):
		if self.conv:
			out = x.unsqueeze(1)
			out = F.relu(self.conv1(out))
			out = F.relu(self.conv2(out))
			out = F.relu(self.conv3(out))
			out = out.view(out.size(0), -1)
		out = torch.cat((out, a), dim=-1)
		out = F.relu(self.fc(out))
		return self.pred_out(out).reshape(out.size(0), 
			self.input_h, self.input_h), self.rew_out(out)

class ModelTrainer:
	def __init__(self, data, model, args):
		self.model = model
		self.args = args

		self.prev_obs = torch.cat([d['prev_obs'] for d in data], dim=0)
		self.prev_act = torch.cat([d['prev_act'] for d in data], dim=0)
		self.obs = torch.cat([d['obs'] for d in data], dim=0)
		self.obs_tp1 = torch.cat([d['next_obs'] for d in data], dim=0)
		self.act = torch.cat([d['act'] for d in data], dim=0)
		self.rew = torch.cat([d['rew'] for d in data], dim=0)
		self.done = torch.cat([d['done'] for d in data], dim=0)

		self.N = len(self.obs)
		self.T = self.args.epochs
		self.device = self.args.device
		self.train_idx = np.random.choice(np.arange(self.N), size=int(0.8 * self.N), replace=False)
		self.val_idx = np.delete(np.arange(self.N), self.train_idx)

		self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.l2_pen)
		self.bs = LinearSchedule(schedule_timesteps=self.args.epochs, final_p=self.args.beta, initial_p=0.0)
		self.tbs = ConstantSchedule(schedule_timesteps=self.args.epochs, p=0.0)

	def _get_loss(self, batch_idx, beta, epoch):
		obs = self.obs[batch_idx].to(self.device)
		act = self.act[batch_idx].to(self.device)
		obs_tp1 = self.obs_tp1[batch_idx].to(self.device)
		rew = self.rew[batch_idx].to(self.device)
		done = self.done[batch_idx].to(self.device)
		prev_obs = self.prev_obs[batch_idx].to(self.device)
		prev_act = self.prev_act[batch_idx].to(self.device)

		with torch.no_grad():
			prev_obs_pred, _ = self.model(prev_obs, prev_act)
		mix_batch_size = int(beta * len(prev_obs_pred))

		if mix_batch_size > 1:
			M = np.random.choice(np.arange(1, len(prev_obs_pred)), size=mix_batch_size, replace=False)
			N = np.random.choice(np.arange(len(obs)), size=len(obs) - mix_batch_size, replace=False)

			pred, r_pred = model(obs[N], act[N])
			pred_prime, r_pred_prime = self.model(prev_obs_pred[M], act[M])

			tot_pred = torch.cat((pred, pred_prime), dim=0)
			tot_r_pred = torch.cat((r_pred, r_pred_prime), dim=0)

			tot_target = torch.cat((obs_tp1[N], obs_tp1[M]), dim=0)
			tot_r_target = torch.cat((rew[N], rew[M]), dim=0)

			pred_loss = F.smooth_l1_loss(tot_pred, tot_target)
			rew_loss = F.binary_cross_entropy_with_logits(tot_r_pred.squeeze(), tot_r_target)
		else:
			pred, r_pred = self.model(obs, act)
			if epoch == 9:
				import pdb; pdb.set_trace()
			# computing losses
			pred_loss = F.smooth_l1_loss(pred, obs_tp1)
			rew_loss = F.binary_cross_entropy_with_logits(r_pred.squeeze(), rew)
		return pred_loss + rew_loss
		
	def train(self):
		self.model.train()
		for t in range(self.T):
			num_batches = self.train_idx.shape[0] // self.args.batch_size
			avg_train_loss = []
			for _ in tqdm(range(num_batches)):
				self.optim.zero_grad()
				batch_idx = np.random.choice(self.train_idx, size=self.args.batch_size, replace=False)
				loss = self._get_loss(batch_idx, self.bs.value(t), t)
				loss.backward()
				avg_train_loss.append(loss.item())
				self.optim.step()

			# validation
			self.model.eval()
			with torch.no_grad():
				batch_idx = np.random.choice(self.val_idx, size=1000, replace=False)
				avg_t_loss = self._get_loss(batch_idx, self.tbs.value(t), t)
			print('====> Epoch: {} TrainLoss: {:.4f} TestLoss: {:.4f}'.format(t, np.mean(avg_train_loss), avg_t_loss.item()))

	def save(self):
		with open(os.path.join(self.args.outdir, 'final.pt'), 'wb') as f:
			torch.save(self.model.state_dict(), f)

def main(args):
	env = gym.make('GridGameTrain-v0')
	obs_shape = env.observation_space.shape
	act_dim = env.action_space.n

	if not os.path.exists(args.outdir):
		os.makedirs(args.outdir + '/', exist_ok=True)

	input_size = obs_shape[0] * obs_shape[1]
	model = Model(input_size, act_dim, conv=args.conv)
	model.to(args.device)

	if not os.path.exists('data.pkl'):
		data = collect_batch_episodes(env, test_eps=args.T, conv=args.conv) # (T, L, hidden)
		pickle.dump(data, open('data.pkl', 'wb'))
	else:
		data = pickle.load(open('data.pkl', 'rb'))

	model_trainer = ModelTrainer(data, model, args)
	model_trainer.train()
	model_trainer.save()
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--conv', action='store_true')
	parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
	parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
	parser.add_argument('--T', type=int, default=5000, help='number of rollouts to collect')
	parser.add_argument('--outdir', type=str, default='mdn_debug/', help='where to save results')
	parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
	parser.add_argument('--l2_pen', type=float, default=1e-9, help='l2 regularization penalty')
	parser.add_argument('--beta', type=float, default=0.0, help='mixing coefficient for data')
	parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
	args = parser.parse_args()

	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
	args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	main(args)
