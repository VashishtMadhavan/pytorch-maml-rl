"""

Generalization Test for Experiments. Will loop through checkpoints and evaluate them on OriginalGame-v0
This will tell us which has best generalization and thus enforce early stopping.
Also helps us understand how generalization differs through training


"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_color_codes(palette='deep')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str)
    return parser.parse_args()

def main(args):
    save_dir = args.logdir + '/saves/'
    ckpt_files = [save_dir + x for x in os.listdir(save_dir) if '.pt' in x]
    for ck in ckpt_files:
        cmd_ = 'python test_custom.py --env OriginalGame-v0 --test-eps 30 --save '
        cmd_ += '--checkpoint {}'.format(ck)
        os.system(cmd_)
    print("Done Testing Checkpoints...")

    log_files = ['test_saves/' + x for x in os.listdir('test_saves/')]
    rewards = []; batches = []
    for l in log_files:
        batch = int(l.split('/').split('.txt')[0])
        lines = [x.rstrip().split() for x in open(l).readlines()]
        batches.append(batch)
        rewards.append(float(lines[0][1]))

    plt.title("Generalization Performance")
    plt.ylabel("MeanReward")
    plt.xlabel("Batch")
    plt.scatter(batches, rewards, c='r')
    plt.savefig('generalization_scatter.png')

if __name__ == "__main__":
    args = parse_args()
    main(args)