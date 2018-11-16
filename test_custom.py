import gym
import numpy as np
import torch
import argparse
import torch

import maml_rl.envs
from maml_rl.policies.conv_lstm_policy import ConvLSTMPolicy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default='CustomGame-v0')
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--random", action="store_true")
    return parser.parse_args()


def load_params(policy_path, env, device):
    policy = ConvLSTMPolicy(
        input_size=env.observation_space.shape,
        output_size=env.action_space.n)
    if device == 'cpu':
        policy.load_state_dict(torch.load(policy_path, map_location=device))
    else:
        policy.load_state_dict(torch.load(policy_path))
    return policy


def evaluate(env, policy, device, test_eps=10, render=False, random=False):
    num_actions = env.action_space.n
    ep_rews = []; ep_steps = []
    for t in range(test_eps):
        obs = env.reset(); done = False
        embed_tensor = torch.zeros(1, num_actions + 2).to(device=device)
        hx = torch.zeros(1, 256).to(device=device)
        cx = torch.zeros(1, 256).to(device=device)
        total_rew = 0; tstep = 0

        while not done:
            if render:
                env.render()
            obs_tensor = torch.from_numpy(np.array(obs)[None]).to(device=device)
            action_dist, value_tensor, hx, cx = policy(obs_tensor, hx, cx, embed_tensor)
            action = action_dist.sample().cpu().numpy()

            if random:
                obs, rew, done, _ = env.step(env.action_space.sample())
            else:
                obs, rew, done, _ = env.step(action[0])

            embed_temp = np.append(np.zeros(num_actions), [rew, float(done)])
            embed_temp[action[0]] = 1.
            embed_tensor = torch.from_numpy(embed_temp[None]).float().to(device=device)

            total_rew += rew; tstep += 1
        ep_rews.append(total_rew); ep_steps.append(tstep)
    return ep_rews, ep_steps
 
def main():
    args = parse_args()
    env = gym.make(args.env_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = load_params(args.checkpoint, env, device)
    episode_rew, episode_steps = evaluate(env, policy, device, render=args.render, random=args.random)

    print("MeanRewards: ",  np.mean(episode_rew))
    print("MeanSteps: ", np.mean(episode_steps))


if __name__ == '__main__':
    main()
