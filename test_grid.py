import gym
import sys; import os
import numpy as np
import torch
import argparse
import imageio
from tqdm import tqdm

import maml_rl.envs
from maml_rl.policies import FFPolicy, GRUPolicy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='GridGameTrain-v0')
    parser.add_argument("--test-eps", type=int, default=1000)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--lstm", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()

def load_params(policy_path, env, device, lstm=False, D=1, N=1):
    if lstm:
        policy = GRUPolicy(env.observation_space.shape[0], env.action_space.n, lstm_size=32, D=D, N=N)
    else:
        policy = FFPolicy(env.observation_space.shape[0], env.action_space.n, D=D)
    policy.load_state_dict(torch.load(policy_path, 
    	map_location=device if device == 'cpu' else None))
    return policy

def evaluate(env, policy, device, test_eps=10, greedy=False, render=False, lstm=False, D=1):
    nA = env.action_space.n; frames = []
    epR = []; epT = []; sepR = []
    for t in tqdm(range(test_eps)):
        obs = env.reset(); done = False
        if lstm:
            hx = torch.zeros(D, 1, 32, 2, 2).to(device=device)
        e_tensor = torch.zeros(1, nA + 2).to(device=device)
        e_tensor[:, 0] = 1.
        R = 0; T = 0

        while not done:
            frames.append(np.array(obs))
            if render: env.render()
            obs_tensor = torch.from_numpy(np.array(obs)[None]).to(device=device)
            if lstm:
                action_dist, value_tensor, hx = policy(obs_tensor, hx, e_tensor)
            else:
                action_dist, value_tensor = policy(obs_tensor)

            if greedy:
                probs = action_dist.probs.detach().cpu().numpy()
                action = np.argmax(probs, axis=1)[0]
            else:
                action = action_dist.sample().cpu().numpy()[0]
            obs, rew, done, info = env.step(action)

            e = np.zeros(nA + 2); e[action] = 1.; e[-2] = rew; e[-1] = float(done)
            e_tensor = torch.from_numpy(e[None]).float().to(device=device)
            R += rew; T += 1
        sepR.append(rew)
        epR.append(R); epT.append(T)
    return epR, epT, np.sign(sepR)
 
def main():
    args = parse_args()
    env = gym.make(args.env)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = load_params(args.checkpoint, env, device, lstm=args.lstm, D=args.d, N=args.n)
    episode_rew, episode_steps, second_ep_rew = evaluate(env, policy, device, greedy=args.greedy,
        test_eps=args.test_eps, render=args.render, lstm=args.lstm, D=args.d)

    if args.save:
        batch = args.checkpoint.split('/')[-1].split('.')[0].split('-')[1]
        save_dir = 'test_saves/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        f = open('{}/{}.txt'.format(save_dir, batch), 'a')
    else:
        f = sys.stdout
    print("MeanRew: {}".format(np.mean(episode_rew)), file=f)
    print("Std.Rew: {}".format(np.std(episode_rew) / np.sqrt(len(episode_rew))), file=f)
    print("MeanT: {}".format(np.mean(episode_steps)), file=f)


if __name__ == '__main__':
    main()
