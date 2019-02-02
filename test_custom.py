import gym
import sys; import os
import numpy as np
import torch
import argparse
import imageio
from tqdm import tqdm

import maml_rl.envs
from maml_rl.policies import ConvLSTMPolicy, ConvCLSTMPolicy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='CustomGame-v0')
    parser.add_argument("--test-eps", type=int, default=10)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--clstm", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


def load_params(policy_path, env, device, clstm=False):
    if not clstm:
        policy = ConvLSTMPolicy(
            input_size=env.observation_space.shape,
            output_size=env.action_space.n)
    else:
        policy = ConvCLSTMPolicy(
            input_size=env.observation_space.shape,
            output_size=env.action_space.n)
    if device == 'cpu':
        policy.load_state_dict(torch.load(policy_path, map_location=device))
    else:
        policy.load_state_dict(torch.load(policy_path))
    return policy


def evaluate(env, policy, device, test_eps=10, greedy=False, render=False, random=False, record=False, clstm=False):
    num_actions = env.action_space.n; total_frames = []
    ep_rews = []; ep_steps = []
    for t in tqdm(range(test_eps)):
        obs = env.reset(); done = False
        embed_tensor = torch.zeros(1, num_actions + 2).to(device=device)
        embed_tensor[:, 0] = 1.
        if not clstm:
            hx = torch.zeros(1, 256).to(device=device)
            cx = torch.zeros(1, 256).to(device=device)
        else:
            hx = torch.zeros(1, 32, 7, 7).to(device=device)
            cx = torch.zeros(1, 32, 7, 7).to(device=device)
        total_rew = 0; tstep = 0

        while not done:
            total_frames.append(np.array(obs))
            if render:
                env.render()
            obs_tensor = torch.from_numpy(np.array(obs)[None]).to(device=device)
            action_dist, value_tensor, hx, cx = policy(obs_tensor, hx, cx, embed_tensor)
            if greedy:
                probs = action_dist.probs.detach().cpu().numpy()
                action = np.argmax(probs, axis=1)
            else:
                action = action_dist.sample().cpu().numpy()

            if random:
                obs, rew, done, _ = env.step(env.action_space.sample())
            else:
                obs, rew, done, _ = env.step(action[0])

            embed_arr = np.zeros(num_actions + 2)
            embed_arr[action[0]] = 1.
            embed_arr[-2] = rew
            embed_arr[-1] = float(done)
            embed_tensor = torch.from_numpy(embed_arr[None]).float().to(device=device)

            total_rew += rew; tstep += 1
        ep_rews.append(total_rew); ep_steps.append(tstep)
    if record:
        frames = [t[:,:,-1] for t in total_frames]
        imageio.mimsave("movie.gif", frames)
    return ep_rews, ep_steps
 
def main():
    args = parse_args()
    env = gym.make(args.env)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = load_params(args.checkpoint, env, device, clstm=args.clstm)
    episode_rew, episode_steps = evaluate(env, policy, device, greedy=args.greedy, test_eps=args.test_eps,
        render=args.render, random=args.random, record=args.record, clstm=args.clstm)

    if args.save:
        batch = args.checkpoint.split('/')[-1].split('.')[0].split('-')[1]
        save_dir = 'test_saves/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        f = open('{}/{}.txt'.format(save_dir, batch), 'a')
    else:
        f = sys.stdout
    print("MeanRewards: {}".format(np.mean(episode_rew)), file=f)
    print("Std.Rewards: {}".format(np.std(episode_rew) / np.sqrt(len(episode_rew))), file=f)
    print("MeanSteps: {}".format(np.mean(episode_steps)), file=f)


if __name__ == '__main__':
    main()
