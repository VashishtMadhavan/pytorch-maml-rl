import maml_rl.envs
import gym
import numpy as np
import torch
import time
import os
import json
from maml_rl.grid_learner import GridLearner
from torch.optim.lr_scheduler import LambdaLR
from maml_rl.utils import gen_utils, torch_utils

def main(args):
    logger = gen_utils.Logger(args.outdir)
    logger.save_config(args)

    learner = GridLearner(env_name=args.env, batch_size=args.batch_size, ent_coef=args.ent_coef,
        num_workers=args.workers, num_batches=args.train_iters, gamma=args.gamma, lr=args.lr,
        tau=args.tau, vf_coef=args.vf_coef, device=args.device, D=args.d, N=args.n)

    logger.set_keys(['MeanReward:','Batch:','Tsteps:','TimePerBatch:'])
    batch = 0
    scheduler = LambdaLR(learner.optimizer, lambda x: (1.0 - x / float(args.train_iters)))

    # Training Loop
    while batch < args.train_iters:
        scheduler.step()
        tstart = time.time()
        episodes = learner.sample()
        learner.surrogate_step(episodes)

        batch_step_time = time.time() - tstart
        tot_rew = torch_utils.total_rewards([episodes.rewards])
        tsteps = (batch + 1) * args.batch_size * 100

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
    if args.hdfs:
        gen_utils.hdfs_save('/ailabs/vash/grid_game/', args.outdir + '/')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Grid learning with LSTMs')

    # General
    parser.add_argument('--hdfs', action='store_false')
    parser.add_argument('--env', type=str, default='GridGameTrain-v0')
    parser.add_argument('--d', type=int, default=1)
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.95, help='discount factor for GAE')
    parser.add_argument('--vf_coef', type=float, default=0.5, help='value function coeff')
    parser.add_argument('--ent_coef', type=float, default=0.01, help='entropy bonus coeff')
    parser.add_argument('--batch-size', type=int, default=200, help='num episodes for gradient est.')
    parser.add_argument('--train-iters', type=int, default=1000, help='training iterations')

    # Miscellaneous
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--outdir', type=str, default='grid_debug')
    parser.add_argument('--workers', type=int, default=200, help='num workers for traj sampling')
    args = parser.parse_args()

    # Device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args)
