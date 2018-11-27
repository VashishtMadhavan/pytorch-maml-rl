import gym
import numpy as np
import torch
import argparse
import torch
import imageio

import maml_rl.envs
from maml_rl.policies.conv_lstm_policy import ConvLSTMPolicy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default='OriginalGame-v0')
    parser.add_argument("--test-eps", type=int, default=10)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--record", action="store_true")
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


def evaluate(env, policy, device, test_eps=10, render=False, random=False, record=False):
    num_actions = env.action_space.n; total_frames = []
    ep_rews = []; ep_steps = []
    for t in range(test_eps):
        env.unwrapped.reset_task(None)
        obs = env.reset(); done = False
        action_embed_tensor = torch.zeros(1, num_actions).to(device=device)
        action_embed_tensor[:, 0] = 1.
        rew_embed_tensor = torch.zeros(1, 2).to(device=device)
        hx = torch.zeros(1, 256).to(device=device)
        cx = torch.zeros(1, 256).to(device=device)
        total_rew = 0; tstep = 0

        while not done:
            total_frames.append(np.array(obs))
            if render:
                env.render()
            obs_tensor = torch.from_numpy(np.array(obs)[None]).to(device=device)
            action_dist, value_tensor, hx, cx = policy(obs_tensor, hx, cx, action_embed_tensor, rew_embed_tensor)
            action = action_dist.sample().cpu().numpy()

            if random:
                obs, rew, done, _ = env.step(env.action_space.sample())
            else:
                obs, rew, done, _ = env.step(action[0])

            action_embed_temp = np.zeros(num_actions)
            action_embed_temp[action[0]] = 1.
            action_embed_tensor = torch.from_numpy(action_embed_temp[None]).float().to(device=device)

            rew_embed_tensor = torch.from_numpy(np.array([rew, float(done)])[None]).float().to(device=device)
            total_rew += rew; tstep += 1
        ep_rews.append(total_rew); ep_steps.append(tstep)
    if record:
        frames = [t[:,:,-1] for t in total_frames]
        imageio.mimsave("movie.gif", frames)
    return ep_rews, ep_steps
 
def main():
    args = parse_args()
    env = gym.make(args.env_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = load_params(args.checkpoint, env, device)
    episode_rew, episode_steps = evaluate(env, policy, device, test_eps=args.test_eps, render=args.render, random=args.random, record=args.record)

    print("MeanRewards: ",  np.mean(episode_rew))
    print("MeanSteps: ", np.mean(episode_steps))


if __name__ == '__main__':
    main()
