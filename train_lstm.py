import maml_rl.envs
import gym
import numpy as np
import torch
import json
from tqdm import tqdm

from maml_rl.lstm_learner import LSTMLearner
from tensorboardX import SummaryWriter

def total_rewards(episodes_rewards, aggregation=torch.mean):
    rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards, dim=0))
        for rewards in episodes_rewards], dim=0))
    return rewards.item()

def main(args):
    writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
    save_folder = './saves/{0}'.format(args.output_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(os.path.join(save_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)

    learner = LSTMLearner(env_name=args.env_name, batch_size=args.batch_size,
        num_workers=args.num_workers, gamma=args.gamma,
        lr=args.lr, tau=args.tau, vf_coef=0.5, device=args.device)
    """
    Training Loop
    """
    for batch in tqdm(range(args.num_batches)):
        episodes = learner.sample()
        learner.step(episodes)

        # Tensorboard
        writer.add_scalar('total_rewards/reward',
            total_rewards([episodes.rewards]), batch)

        # Save policy network
        with open(os.path.join(save_folder,
                'policy-{0}.pt'.format(batch)), 'wb') as f:
            torch.save(learner.policy.state_dict(), f)


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Reinforcement learning with LSTMs')

    # General
    parser.add_argument('--env-name', type=str,
        help='name of the environment')
    parser.add_argument('--gamma', type=float, default=0.99,
        help='value of the discount factor gamma')
    parser.add_argument('--tau', type=float, default=1.0,
        help='value of the discount factor for GAE')
    parser.add_argument('--vf_coef', type=float, default=0.5,
        help='coefficient for value function portion of loss')
    parser.add_argument('--batch-size', type=int, default=40,
        help='number of episodes to estimate gradient')
    parser.add_argument('--lr', type=float, default=1e-4,
        help='learning rate for the LSTM network')
    parser.add_argument('--num-batches', type=int, default=1000,
        help='number of batches')    

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='lstm',
        help='name of the output folder')
    parser.add_argument('--num-workers', type=int, default=20,
        help='number of workers for trajectories sampling')
    parser.add_argument('--device', type=str, default='cpu',
        help='set the device (cpu or cuda)')

    args = parser.parse_args()
    # Create logs and saves folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./saves'):
        os.makedirs('./saves')
    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])

    main(args)
