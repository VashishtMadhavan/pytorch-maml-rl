import maml_rl.envs
import gym
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

    learner = LSTMLearner(env_name=args.env, batch_size=args.batch_size, ent_coef=args.ent_coef, latent=args.latent_model,
        D=args.D, N=args.N, num_workers=args.workers, num_batches=args.train_iters, gamma=args.gamma, use_bn=args.use_bn,
        cnn_type=args.cnn_type, lr=args.lr, tau=args.tau, vf_coef=args.vf_coef, l2_coef=args.l2_coef, device=args.device, clstm=args.clstm)

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
        if args.ppo:
            # PPO step
            learner.surrogate_step(episodes)
        else:
            # Regular A2C step
            learner.step(episodes)
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
        gen_utils.hdfs_save('/ailabs/vash/custom_game/', args.outdir + '/')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Reinforcement learning with LSTMs')

    # General
    parser.add_argument('--env', type=str)
    parser.add_argument('--ppo', action='store_true')
    parser.add_argument('--hdfs', action='store_false')
    parser.add_argument('--clstm', action='store_true', help='whether or not to use a conv-lstm')
    parser.add_argument('--load', action='store_true', help='loading previous experiment')

    parser.add_argument('--latent_model', type=str, default=None, help='model for DARLA style training')
    parser.add_argument('--D', type=int, default=1, help='stack depth of LSTMs')
    parser.add_argument('--N', type=int, default=1, help='number of repeated LSTM steps before output')
    parser.add_argument('--cnn_type', type=str, default='nature', help='which type of network encoder to use: (nature, impala)')
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.95, help='discount factor for GAE')
    parser.add_argument('--vf_coef', type=float, default=0.5, help='value function coeff')
    parser.add_argument('--ent_coef', type=float, default=0.05, help='entropy bonus coeff')
    parser.add_argument('--l2_coef', type=float, default=0., help='L2 regularization coeff')
    parser.add_argument('--use_bn', action='store_true', help='use batch normalizaton')
    parser.add_argument('--batch-size', type=int, default=100, help='num episodes for gradient est.')
    parser.add_argument('--train-iters', type=int, default=5000, help='training iterations')

    # Miscellaneous
    parser.add_argument('--outdir', type=str, default='debug')
    parser.add_argument('--workers', type=int, default=80, help='num workers for traj sampling')
    args = parser.parse_args()

    # Device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main(args)
