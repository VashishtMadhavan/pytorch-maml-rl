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

class FFModel(nn.Module):
	def __init__(self, input_size, action_dim, hidden_size, K=2):
		super(FFModel, self).__init__()
		self.input_size = input_size
		self.action_dim = action_dim
		self.hidden_size = hidden_size
		self.K = K # number of gaussians in GMM

		self.fc = nn.Linear(self.input_size + self.action_dim, self.hidden_size)
		self.pred_out = nn.Linear(self.hidden_size, self.input_size)
		self.rew_out = nn.Linear(self.hidden_size, 1) # predict either reward of 1 or 0

	def forward(self, x, a):
		out = torch.cat((x, a), dim=-1)
		out = F.relu(self.fc(out))
		return self.pred_out(out), self.rew_out(out)

class ConvModel(nn.Module):
	def __init__(self, input_size, action_dim, hidden_size, K=2):
		super(ConvModel, self).__init__()
		self.input_size = input_size
		self.action_dim = action_dim
		self.hidden_size = hidden_size
		self.K = K

		self.conv1 = nn.Conv2d(1, hidden_size, kernel_size=3, stride=1)
		self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1)
		self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1)
		self.fc = nn.Linear(hidden_size + action_dim, 32)
		self.rew_out = nn.Linear(32, 1)

		self.deconv1 = nn.ConvTranspose2d(32, hidden_size, kernel_size=3, stride=1)
		self.deconv2 = nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=3, stride=1)
		self.deconv2 = nn.ConvTranspose2d(hidden_size, 1, kernel_size=3, stride=1)

	def forward(self, x, a):
		out = x.unsqueeze(1)
		out = F.relu(self.conv1(out))
		out = F.relu(self.conv2(out))
		out = F.relu(self.conv3(out))

		out = out.view(out.size(0), -1)
		out = torch.cat((out, a), dim=-1)
		out = F.relu(self.fc(out))
		rew_pred = self.rew_out(out)

		out = out.unsqueeze(-1).unsqueeze(-1)
		out = F.relu(self.deconv1(out))
		out = F.relu(self.deconv2(out))
		pred = self.deconv3(out)
		return pred.squeeze(), rew_pred

def get_batch(data, batch_size, device=torch.device('cpu')):
	random_idx = np.random.choice(np.arange(len(data)), size=batch_size, replace=False)
	data_dicts = [data[r] for r in random_idx]

	prev_obs_batch = torch.cat([d['prev_obs'] for d in data_dicts], dim=0)
	prev_act_batch = torch.cat([d['prev_act'] for d in data_dicts], dim=0)
	obs_batch = torch.cat([d['obs'] for d in data_dicts], dim=0)
	obs_tp1_batch = torch.cat([d['next_obs'] for d in data_dicts], dim=0)
	act_batch = torch.cat([d['act'] for d in data_dicts], dim=0)
	rew_batch = torch.cat([d['rew'] for d in data_dicts], dim=0)
	done_batch = torch.cat([d['done'] for d in data_dicts], dim=0)
	return obs_batch.to(device), act_batch.to(device), obs_tp1_batch.to(device), \
		rew_batch.to(device), done_batch.to(device), prev_obs_batch.to(device), prev_act_batch.to(device)

def get_ff_loss(obs, act, obs_tp1, rew, done, prev_obs, prev_act, model, epoch, beta_schedule):
	with torch.no_grad():
		prev_obs_pred, _ = model(prev_obs, prev_act)
	pred_frac = beta_schedule.value(epoch)
	mix_batch_size = int(pred_frac * len(prev_obs_pred))

	if mix_batch_size > 1:
		M = np.random.choice(np.arange(1, len(prev_obs_pred)), size=mix_batch_size, replace=False)
		N = np.random.choice(np.arange(len(obs)), size=len(obs) - mix_batch_size, replace=False)

		pred, r_pred = model(obs[N], act[N])
		pred_prime, r_pred_prime = model(prev_obs_pred[M], act[M])

		tot_pred = torch.cat((pred, pred_prime), dim=0)
		tot_r_pred = torch.cat((r_pred, r_pred_prime), dim=0)

		tot_target = torch.cat((obs_tp1[N], obs_tp1[M]), dim=0)
		tot_r_target = torch.cat((rew[N], rew[M]), dim=0)

		pred_loss = F.smooth_l1_loss(tot_pred, tot_target)
		rew_loss = F.binary_cross_entropy_with_logits(tot_r_pred.squeeze(), tot_r_target)
	else:
		pred, r_pred = model(obs, act)
		# computing losses
		pred_loss = F.smooth_l1_loss(pred, obs_tp1)
		rew_loss = F.binary_cross_entropy_with_logits(r_pred.squeeze(), rew)
	return pred_loss + rew_loss


def main(args):
	env = gym.make('GridGameTrain-v0')
	obs_shape = env.observation_space.shape
	act_dim = env.action_space.n
	# creating output dir
	if not os.path.exists(args.outdir):
		os.makedirs(args.outdir + '/', exist_ok=True)

	if not args.conv:
		model = FFModel(input_size=obs_shape[0] * obs_shape[1], action_dim=act_dim, hidden_size=args.hidden_size)
	else:
		model = ConvModel(input_size=obs_shape[0] * obs_shape[1], action_dim=act_dim, hidden_size=args.hidden_size)
	model.to(args.device)
	loss_fn = get_ff_loss

	if not os.path.exists('data.pkl'):
		data = collect_batch_episodes(env, test_eps=args.T, conv=args.conv) # (T, L, hidden)
		pickle.dump(data, open('data.pkl', 'wb'))
	else:
		data = pickle.load(open('data.pkl', 'rb'))

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_pen)

	random.shuffle(data)
	train_data, test_data = data[:int(0.8 * args.T)], data[int(0.8 * args.T):]
	beta_schedule = LinearSchedule(schedule_timesteps=args.epochs, final_p=args.beta, initial_p=0.0)
	test_beta_schedule = ConstantSchedule(schedule_timesteps=args.epochs, p=0.0)

	for ep in range(args.epochs):
		# Training
		model.train()
		num_batches = len(train_data) // args.batch_size
		avg_train_loss = []
		for idx in tqdm(range(num_batches)):
			obs_batch, act_batch, obs_tp1_batch, rew_batch, done_batch, prev_obs_batch, prev_act_batch = get_batch(train_data, args.batch_size, device=args.device)
			optimizer.zero_grad()
			loss = loss_fn(obs_batch, act_batch, obs_tp1_batch, rew_batch, done_batch, prev_obs_batch, prev_act_batch, model, ep, beta_schedule=beta_schedule)
			loss.backward()
			avg_train_loss.append(loss.item())
			optimizer.step()

		model.eval()
		t_obs_batch, t_act_batch, t_obs_tp1_batch, t_rew_batch, t_done_batch, t_prev_obs_batch, t_prev_act_batch = get_batch(test_data, 1000, device=args.device)
		with torch.no_grad():
			avg_t_loss = loss_fn(t_obs_batch, t_act_batch, t_obs_tp1_batch, t_rew_batch, t_done_batch, t_prev_obs_batch, t_prev_act_batch, model, ep, beta_schedule=test_beta_schedule)
		print('====> Epoch: {} TrainLoss: {:.4f} TestLoss: {:.4f}'.format(ep, np.mean(avg_train_loss), avg_t_loss.item()))

	with open(os.path.join(args.outdir, 'final.pt'), 'wb') as f:
		torch.save(model.state_dict(), f)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--conv', action='store_true')
	parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
	parser.add_argument('--hidden_size', type=int, default=16, help='hidden size for layers')
	parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
	parser.add_argument('--T', type=int, default=20000, help='number of rollouts to collect')
	parser.add_argument('--outdir', type=str, default='mdn_debug/', help='where to save results')
	parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
	parser.add_argument('--l2_pen', type=float, default=1e-9, help='l2 regularization penalty')
	parser.add_argument('--beta', type=float, default=0.0, help='mixing coefficient for data')
	parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
	args = parser.parse_args()

	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
	args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	main(args)
