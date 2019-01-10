import maml_rl.envs
import gym
import numpy as np
import torch
import time
import json
from maml_rl.lstm_learner import LSTMLearner

def hdfs_save(hdfs_dir, filename):
    os.system('/opt/hadoop/latest/bin/hdfs dfs -copyFromLocal -f {} {}'.format(filename, hdfs_dir))

def total_rewards(episodes_rewards, aggregation=torch.mean):
    rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards, dim=0))
        for rewards in episodes_rewards], dim=0))
    return rewards.item()

def main(args):
    save_folder = '{0}/saves/'.format(args.outdir)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with open(os.path.join(save_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)

    learner = LSTMLearner(env_name=args.env, batch_size=args.batch_size,
        num_workers=args.workers, num_batches=args.train_iters, gamma=args.gamma,
        lr=args.lr, tau=args.tau, vf_coef=args.vf_coef, device=args.device)

    """
    Training Loop
    """
    for batch in range(args.train_iters):
        tstart = time.time()
        episodes = learner.sample()
        if args.ppo:
            # PPO step
            learner.surrogate_step(episodes)
        else:
            # Regular A2C step
            learner.step(episodes)
        batch_step_time = time.time() - tstart

        # Writing Episode Rewards
        tot_rew = total_rewards([episodes.rewards])
        tsteps = (batch + 1) * args.batch_size * episodes.observations.size(0)
        print("MeanReward: {0} Batch: {1} Tsteps: {2} TimePerBatch: {3}".format(tot_rew, batch, tsteps, batch_step_time))

        with open('{0}/log.txt'.format(args.outdir), 'a') as f:
            print("MeanReward: {0} Batch: {1} Tsteps: {2} TimePerBatch: {3}".format(tot_rew, batch, tsteps, batch_step_time), file=f)

        # Save policy network
        if batch % 50 == 0:
            with open(os.path.join(save_folder, 'policy-{0}.pt'.format(batch)), 'wb') as f:
                torch.save(learner.policy.state_dict(), f)

    learner.envs.close()
    if args.hdfs:
        hdfs_save('/ailabs/vash/custom_game/', args.outdir + '/')

if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Reinforcement learning with LSTMs')

    # General
    parser.add_argument('--env', type=str)
    parser.add_argument('--ppo', action='store_true')
    parser.add_argument('--hdfs', action='store_false')
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.95, help='discount factor for GAE')
    parser.add_argument('--vf_coef', type=float, default=0.5, help='value function coeff')
    parser.add_argument('--batch-size', type=int, default=240, help='num episodes for gradient est.')
    parser.add_argument('--train-iters', type=int, default=500, help='training iterations')

    # Miscellaneous
    parser.add_argument('--outdir', type=str, default='debug')
    parser.add_argument('--workers', type=int, default=60, help='num workers for traj sampling')
    args = parser.parse_args()

    # Device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.outdir += '-{0}'.format(os.environ['SLURM_JOB_ID'])

    main(args)
