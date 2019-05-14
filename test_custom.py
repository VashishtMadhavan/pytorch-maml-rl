import gym
import sys; import os
import numpy as np
import torch
import argparse
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt
import maml_rl.envs
from maml_rl.policies import ConvGRUPolicy, ConvCGRUPolicy, ConvLSTMPolicy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='CustomGame-v0')
    parser.add_argument("--test-eps", type=int, default=10)
    parser.add_argument("--checkpoint", type=str)

    parser.add_argument("--D", type=int, default=1)
    parser.add_argument("--N", type=int, default=1)
    parser.add_argument("--cnn_type", type=str, default='nature')
    parser.add_argument("--clstm", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()

def load_params(policy_path, env, device, clstm=False, cnn_type='nature', D=1, N=1):
    obs_shape = env.observation_space.shape
    a_dim = env.action_space.n
    if not clstm:
        policy = ConvGRUPolicy(obs_shape, a_dim, cnn_type=cnn_type, D=D, N=N)
    else:
        policy = ConvCLSTMPolicy(obs_shape, a_dim, D=D, N=N)
    policy.load_state_dict(torch.load(policy_path, map_location=device if device == 'cpu' else None))
    return policy

def evaluate(env, policy, device, test_eps=10, greedy=False, render=False, random=False, 
            record=False, clstm=False, D=1):
    nA = env.action_space.n; frames = []
    epR = []; epT = []; sepR = []
    for t in tqdm(range(test_eps)):
        obs = env.reset(); done = False
        e_tensor = torch.zeros(1, nA + 2).to(device=device)
        e_tensor[:, 0] = 1.
        if not clstm:
            hx = torch.zeros(policy.D, 1, 256).to(device=device)
        else:
            hx = torch.zeros(policy.D, 1, 32, 7, 7).to(device=device)
        R = 0; T = 0

        while not done:
            frames.append(np.array(obs))
            if render: env.render()
            obs_inp = np.array(obs)[None]
            obs_tensor = torch.from_numpy(obs_inp).to(device=device)
            action_dist, value_tensor, hx = policy(obs_tensor, hx, e_tensor)

            if greedy:
                probs = action_dist.probs.detach().cpu().numpy()
                action = np.argmax(probs, axis=1)[0]
            elif random:
                action = env.action_space.sample()
            else:
                action = action_dist.sample().cpu().numpy()[0]
            obs, rew, done, info = env.step(action)

            if 'v0' in env.spec.id:
                term_flag = float(done)
            else:
                term_flag = np.sign(info['done']) if 'done' in info else 0.0

            e = np.zeros(nA + 2); e[action] = 1.; e[-2] = rew; e[-1] = term_flag
            e_tensor = torch.from_numpy(e[None]).float().to(device=device)
            R += rew; T += 1
        sepR.append(rew)
        epR.append(R); epT.append(T)
    if record:
        imageio.mimsave("movie.gif", frames)
    return epR, epT, np.sign(sepR)
 
def main():
    args = parse_args()
    env = gym.make(args.env)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = load_params(args.checkpoint, env, device, clstm=args.clstm, cnn_type=args.cnn_type, D=args.D, N=args.N)
    policy.to(device)
    episode_rew, episode_steps, second_ep_rew = evaluate(env, policy, device, greedy=args.greedy, test_eps=args.test_eps,
        render=args.render, random=args.random, record=args.record, clstm=args.clstm)

    if 'v1' in env.spec.id:
        episode_rew = np.array(episode_rew); second_ep_rew = np.array(second_ep_rew)
        first_ep_rew = episode_rew - second_ep_rew
        type_err = np.zeros(4)
        type_err[0] = np.sum(episode_rew == 2) / len(episode_rew)
        type_err[3] = np.sum(episode_rew == 0) / len(episode_rew)
        type_err[1] = np.sum(np.clip(first_ep_rew - second_ep_rew, 0, 2)) / len(episode_rew)
        type_err[2] = np.sum(np.clip(second_ep_rew - first_ep_rew, 0, 2)) / len(episode_rew)

        plt.title("Return Distribution per Episode")
        plt.ylabel("'%' of trajectories")
        plt.xticks([1,2,3,4], ['BothRew', 'FirstRew', 'SecondRew', 'NoRew'])
        plt.bar([1,2,3,4],type_err)
        plt.savefig("error_bdown.png")
        print("SecondMeanRew: {}".format(np.mean(second_ep_rew)))
        print("SecondStdRew: {}".format(np.std(second_ep_rew) / np.sqrt(len(second_ep_rew))))

    if args.save:
        batch = args.checkpoint.split('/')[-1].split('.')[0].split('-')[1]
        save_dir = 'test_saves/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        f = open('{}/{}.txt'.format(save_dir, batch), 'a')
    else:
        f = sys.stdout
    print("MeanRewards: {}".format(np.mean(episode_rew)), file=f)
    print("Std.Rewards: {}".format(np.std(episode_rew) / np.sqrt(len(episode_rew))), file=f)
    print("MeanSteps: {}".format(np.mean(episode_steps)), file=f)


if __name__ == '__main__':
    main()
