import gym
import sys; import os
import numpy as np
import torch
import argparse
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt

import maml_rl.envs
from maml_rl.policies import ConvLSTMPolicy, ConvCLSTMPolicy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='CustomGame-v0')
    parser.add_argument("--test-eps", type=int, default=10)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--cnn_type", type=str, default='nature')
    parser.add_argument("--clstm", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


def load_params(policy_path, env, device, clstm=False, cnn_type='nature'):
    if not clstm:
        policy = ConvLSTMPolicy(
            input_size=env.observation_space.shape,
            output_size=env.action_space.n,
            cnn_type=cnn_type)
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
    ep_rews = []; ep_steps = []; second_ep_rews = []
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
                obs, rew, done, info = env.step(env.action_space.sample())
            else:
                obs, rew, done, info = env.step(action[0])

            embed_arr = np.zeros(num_actions + 2)
            embed_arr[action[0]] = 1.
            embed_arr[-2] = rew
            embed_arr[-1] = float(done)
            embed_tensor = torch.from_numpy(embed_arr[None]).float().to(device=device)
            total_rew += rew; tstep += 1
        second_ep_rews.append(rew)
        ep_rews.append(total_rew); ep_steps.append(tstep)
    if record:
        frames = [t[:,:,-1] for t in total_frames]
        imageio.mimsave("movie.gif", frames)
    return ep_rews, ep_steps, second_ep_rews
 
def main():
    args = parse_args()
    env = gym.make(args.env)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = load_params(args.checkpoint, env, device, clstm=args.clstm, cnn_type=args.cnn_type)
    episode_rew, episode_steps, second_ep_rew = evaluate(env, policy, device, greedy=args.greedy, test_eps=args.test_eps,
        render=args.render, random=args.random, record=args.record, clstm=args.clstm)

    if 'v1' in env.spec.id:
        episode_rew = np.array(episode_rew); second_ep_rew = np.array(second_ep_rew)
        first_ep_rew = episode_rew - second_ep_rew
        type_err = np.zeros(4)
        type_err[0] = np.sum(episode_rew == 2) / len(episode_rew)
        type_err[3] = np.sum(episode_rew == 0) / len(episode_rew)
        type_err[1] = np.sum(np.clip(first_ep_rew - second_ep_rew, 0, 2)) / len(episode_rew)
        type_err[2] = np.sum(np.clip(second_ep_rew - first_ep_rew, 0, 2)) / len(episode_rew)

        plt.title("Return Distribution per Episode")
        plt.ylabel("'%' of trajectories")
        plt.xticks([1,2,3,4], ['BothRew', 'FirstRew', 'SecondRew', 'NoRew'])
        plt.bar([1,2,3,4],type_err)
        plt.savefig("error_bdown.png")
        print("SecondMeanRew: {}".format(np.mean(second_ep_rew)))

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
