import maml_rl.envs
import gym
import numpy as np
import torch
import time
import json
from maml_rl.bandit_learner import BanditLearner
from maml_rl.utils import gen_utils, torch_utils

def main(args):
    # TODO: check if logger is necessary here
    logger = gen_utils.Logger(args.outdir)
    logger.save_config(args)

    learner = BanditLearner(k=args.k, n=args.n, batch_size=args.batch_size, ent_coef=args.ent_coef,
        num_workers=args.workers, num_batches=args.train_iters, gamma=args.gamma, lr=args.lr, 
        tau=args.tau, vf_coef=args.vf_coef, device=args.device, D=args.d)

    logger.set_keys(['MeanReward:','Batch:','Tsteps:','TimePerBatch:'])
    batch = 0

    # Training Loop
    while batch < args.train_iters:
        tstart = time.time()
        episodes = learner.sample()
        learner.surrogate_step(episodes)

        batch_step_time = time.time() - tstart
        tot_rew = torch_utils.total_rewards([episodes.rewards])
        tsteps = (batch + 1) * args.batch_size * args.n

        # Logging metrics
        logger.logkv('MeanReward:', tot_rew)
        logger.logkv('Batch:', batch)
        logger.logkv('Tsteps:', tsteps)
        logger.logkv('TimePerBatch:', batch_step_time)
        logger.print_results()

        # Save policy network
        if batch % 50 == 0:
            logger.save_policy(batch, learner.policy)
        batch += 1

    # Saving the final policy
    logger.save_policy(batch, learner.policy)
    learner.envs.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Bandit learning with LSTMs')

    # General
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--d', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.3, help='discount factor for GAE')
    parser.add_argument('--vf_coef', type=float, default=0.05, help='value function coeff')
    parser.add_argument('--ent_coef', type=float, default=0.01, help='entropy bonus coeff')
    parser.add_argument('--batch-size', type=int, default=2500, help='num episodes for gradient est.')
    parser.add_argument('--train-iters', type=int, default=1000, help='training iterations')

    # Miscellaneous
    parser.add_argument('--outdir', type=str, default='bandit_debug')
    parser.add_argument('--workers', type=int, default=100, help='num workers for traj sampling')
    args = parser.parse_args()

    # Device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args)
