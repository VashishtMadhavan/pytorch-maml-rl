import maml_rl.envs
import gym
import numpy as np
import torch
import json
from tqdm import tqdm

from maml_rl.lstm_learner import LSTMLearner

def hdfs_save(hdfs_dir, filename):
    os.system('/opt/hadoop/latest/bin/hdfs dfs -copyFromLocal -f {} {}'.format(filename, hdfs_dir))

def total_rewards(episodes_rewards, aggregation=torch.mean):
    rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards, dim=0))
        for rewards in episodes_rewards], dim=0))
    return rewards.item()

def main(args):
    log_folder = '{0}/logs/'.format(args.output_folder)
    save_folder = '{0}/saves/'.format(args.output_folder)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with open(os.path.join(save_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)

    learner = LSTMLearner(env_name=args.env_name, batch_size=args.batch_size,
        num_workers=args.num_workers, num_batches=args.num_batches, gamma=args.gamma,
        lr=args.lr, tau=args.tau, vf_coef=args.vf_coef, device=args.device)

    with open(os.path.join(log_folder, 'log.txt'), 'a') as f:
        print("EpisodeReward", file=f)

    """
    Training Loop
    """
    for batch in tqdm(range(args.num_batches)):
        episodes = learner.sample()
        learner.step(episodes)

        # Writing Episode Rewards
        tot_rew = total_rewards([episodes.rewards])
        with open(os.path.join(log_folder, 'log.txt'), 'a') as f:
            print('{}'.format(tot_rew), file=f)

        tsteps = (batch + 1) * args.batch_size * 200
        print("Total Rew: {0} Batch: {1}  Timesteps: {2}".format(tot_rew, batch, tsteps))

        # Save policy network
        with open(os.path.join(save_folder, 'policy-{0}.pt'.format(batch)), 'wb') as f:
            torch.save(learner.policy.state_dict(), f)

    learner.envs.close()
    if args.hdfs:
        hdfs_save('/ailabs/vash/custom_game/', args.output_folder + '/')

if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Reinforcement learning with LSTMs')

    # General
    parser.add_argument('--env-name', type=str,
        help='name of the environment')
    parser.add_argument('--hdfs', action='store_false')
    parser.add_argument('--gamma', type=float, default=0.99,
        help='value of the discount factor gamma')
    parser.add_argument('--tau', type=float, default=1.0,
        help='value of the discount factor for GAE')
    parser.add_argument('--vf_coef', type=float, default=0.25,
        help='coefficient for value function portion of loss')
    parser.add_argument('--batch-size', type=int, default=60,
        help='number of episodes to estimate gradient')
    parser.add_argument('--lr', type=float, default=7e-4,
        help='learning rate for the LSTM network')
    parser.add_argument('--num-batches', type=int, default=5000,
        help='number of batches')    

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='debug',
        help='name of the output folder')
    parser.add_argument('--num-workers', type=int, default=60,
        help='number of workers for trajectories sampling')
    parser.add_argument('--device', type=str, default='cpu',
        help='set the device (cpu or cuda)')
    args = parser.parse_args()

    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])

    main(args)
