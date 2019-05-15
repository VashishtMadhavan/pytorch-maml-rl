import maml_rl.envs
import gym
import os
import numpy as np
import torch
import time
import json
from maml_rl.lstm_learner import LSTMLearner
from maml_rl.utils import gen_utils, torch_utils

def main(args):
    if args.load:
        try:
            gen_utils.hdfs_load('/ailabs/vash/custom_game/', args.outdir + '/')
        except:
            print("Cannot find experiment on HDFS, starting from beginning...")

    logger = gen_utils.Logger(args.outdir)
    logger.save_config(args)

    learner = LSTMLearner(env_name=args.env, ent_coef=args.ent_coef, n_step=args.n_step,
        num_workers=args.workers, num_batches=args.train_iters, gamma=args.gamma, lr=args.lr, 
        tau=args.tau, vf_coef=args.vf_coef, device=args.device, clstm=args.clstm)

    # Loading last checkpoint
    if args.load and logger.hdfs_found:
        save_filename, batch = logger.load_checkpoint()
        learner.policy.load_state_dict(torch.load(save_filename))
    else:
        logger.set_keys(['MeanReward:','Batch:','Tsteps:','TimePerBatch:'])
        batch = 0

    # Training Loop
    while batch < args.train_iters:
        tstart = time.time()
        episodes = learner.sample()
        learner.step(episodes)

        batch_step_time = time.time() - tstart
        tsteps = (batch + 1) * args.workers * args.n_step

        # Logging metrics
        logger.logkv('MeanReward:', np.mean(learner.reward_log))
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
        gen_utils.hdfs_save('/ailabs/vash/custom_game/', args.outdir + '/')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Reinforcement learning with LSTMs')

    # General
    parser.add_argument('--env', type=str, default='CustomGame-v0')
    parser.add_argument('--hdfs', action='store_false')
    parser.add_argument('--clstm', action='store_true', help='whether or not to use a conv-lstm')
    parser.add_argument('--load', action='store_true', help='loading previous experiment')

    # Training Args
    parser.add_argument('--lr', type=float, default=2.5e-4, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--tau', type=float, default=0.95, help='discount factor for GAE')
    parser.add_argument('--vf_coef', type=float, default=0.5, help='value function coeff')
    parser.add_argument('--ent_coef', type=float, default=0.05, help='entropy bonus coeff')
    parser.add_argument('--train-iters', type=int, default=5000, help='training iterations')
    parser.add_argument('--workers', type=int, default=128, help='num episodes for gradient est.')
    parser.add_argument('--n_step', type=int, default=128, help='number of steps per PG update')
    
    # Miscellaneous
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--outdir', type=str, default='custom_game_debug')
    args = parser.parse_args()

    # Device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args)
