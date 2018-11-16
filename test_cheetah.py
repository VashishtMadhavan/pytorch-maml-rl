import numpy as np
import torch
import argparse

from maml_rl.envs.mujoco.half_cheetah import HalfCheetahDirEnv
from maml_rl.policies.normal_mlp import NormalMLPPolicy
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler
from maml_rl.metalearner import MetaLearner
from main import total_rewards

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default='HalfCheetahDir-v1')
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--random", action="store_true")
    return parser.parse_args()


def load_meta_learner_params(policy_path, baseline_path, env, device):
    policy = NormalMLPPolicy(
        int(np.prod(env.observation_space.shape)),
        int(np.prod(env.action_space.shape)), 
        hidden_sizes=(100, 100)) # We should actually get this from config
    if device == 'cpu':
        policy.load_state_dict(torch.load(policy_path, map_location=device))
    else:
        policy.load_state_dict(torch.load(policy_path))
    baseline = LinearFeatureBaseline(int(np.prod(env.observation_space.shape)))
    if device == 'cpu':
        baseline.load_state_dict(torch.load(baseline_path, map_location=device))
    else:
        baseline.load_state_dict(torch.load(baseline_path))
    return policy, baseline


def evaluate(env, task, policy, max_path_length=100, render=False, random=False):
    cum_reward = 0
    t = 0
    env.reset_task(task)
    obs = env.reset()
    for _ in range(max_path_length):
        if render:
            env.render()
        obs_tensor = torch.from_numpy(obs).to(device='cpu').type(torch.FloatTensor)
        action_tensor = policy(obs_tensor, params=None).sample()
        action = action_tensor.cpu().numpy()
        if random:
            obs, rew, done, _ = env.step(env.action_space.sample())
        else: 
            obs, rew, done, _ = env.step(action)
        cum_reward += rew
        t += 1
        if done:
            break


    print("========EVAL RESULTS=======")
    print("Return: {}, Timesteps:{}".format(cum_reward, t))
    print("===========================")


def main():
    args = parse_args()
    env = HalfCheetahDirEnv()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sampler = BatchSampler(args.env_name, batch_size=20, num_workers=8)
    baseline_path = args.checkpoint.replace("policy", "baseline")
    policy, baseline = load_meta_learner_params(args.checkpoint, baseline_path, sampler.envs, device=device)
    learner = MetaLearner(sampler, policy, baseline, gamma=0.99, fast_lr=0.1, tau=1.0, device=device)

    tasks = sampler.sample_tasks(num_tasks=40)
    evaluate(env, tasks[0], learner.policy, max_path_length=200, render=args.render, random=args.random)
    episodes = learner.sample(tasks, first_order=False)

    print("TotalPreRewards: ",  total_rewards([ep.rewards for ep, _ in episodes]))
    print("TotalPostRewards: ",  total_rewards([ep.rewards for _, ep in episodes]))
    evaluate(env, tasks[0], learner.policy, max_path_length=200, render=args.render, random=args.random)


if __name__ == '__main__':
    main()