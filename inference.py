import numpy as np
import torch
import argparse

from maml_rl.envs.mujoco.half_cheetah import HalfCheetahDirEnv
from maml_rl.policies.normal_mlp import NormalMLPPolicy
from maml_rl.sampler import BatchSampler
from maml_rl.metalearner import MetaLearner

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default='HalfCheetahDir-v1')
    parser.add_argument("--checkpoint", type=str)
    return parser.parse_args()


def load_meta_learner_params(policy_path, env):
    policy_params = torch.load(policy_path)
    policy = NormalMLPPolicy(
        int(np.prod(env.observation_space.shape)),
        int(np.prod(env.action_space.shape)), 
        hidden_sizes=(100, 100)) # We should actually get this from config
    policy.load_state_dict(policy_params)
    baseline = LinearFeatureBaseline(int(np.prod(env.observation_space.shape)))
    return policy, baseline


def evaluate(env, task, policy, max_path_length=100):
    cum_reward = 0
    t = 0
    env.reset_task(task)
    obs = env.reset()
    for _ in range(max_path_length):
        env.render()
        obs_tensor = torch.from_numpy(obs).to(device='cpu').type(torch.FloatTensor)
        action_tensor = policy(obs_tensor, params=None).sample()
        action = action_tensor.cpu().numpy()
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
    policy, baseline = load_meta_learner_params(args.checkpoint, env)
    sampler = BatchSampler(env_name=args.env_name, batch_size=20, num_workers=2)
    learner = MetaLearner(sampler, policy, baseline)

    tasks = sampler.sample_tasks(num_tasks=1)
    sampler.reset_task(task[0])
    episodes = sampler.sample(policy)
    new_params = learner.adapt(episodes)
    policy.load_state_dict(new_params)
    evaluate(env, task, policy)


if __name__ == '__main__':
    main()