import gym
import sys; import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import imageio
import maml_rl.envs
from tqdm import tqdm
from maml_rl.policies.conv_lstm_policy import ConvLSTMPolicy

from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imresize

searchlight = lambda I, mask: I * mask + gaussian_filter(I, sigma=3) * (1 - mask) # choose an area NOT to blur
occlude = lambda I, mask: I * (1 - mask) + gaussian_filter(I, sigma=3) * mask # choose an area to blur

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='CustomGame-v0')
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()

def get_mask(center, size, r):
    y,x = np.ogrid[-center[0]:size[0]-center[0], -center[1]:size[1]-center[1]]
    keep = x*x + y*y <= 1
    mask = np.zeros(size) ; mask[keep] = 1 # select a circle of pixels
    mask = gaussian_filter(mask, sigma=r) # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
    return mask / mask.max()

def rollout(env, policy, device, render=False):
    history = {'ins': [], 'logits': [], 'values': [], 'outs': [], 'hx': [], 'cx': [], 'embed': []}
    A = env.action_space.n
    obs = env.reset(); done = False
    embed_tensor = torch.zeros(1, A + 2).to(device=device)
    embed_tensor[:, 0] = 1.
    hx = torch.zeros(1, 256).to(device=device)
    cx = torch.zeros(1, 256).to(device=device)

    while not done:
        if render: env.render()
        obs_tensor = torch.from_numpy(np.array(obs)[None]).to(device=device)
        action_dist, value_tensor, hx, cx = policy(obs_tensor, hx, cx, embed_tensor)
        
        action = action_dist.sample().cpu().numpy()
        obs, rew, done, _ = env.step(action[0])

        embed_arr = np.zeros(A + 2)
        embed_arr[action[0]] = 1.
        embed_arr[-2] = rew
        embed_arr[-1] = float(done)
        embed_tensor = torch.from_numpy(embed_arr[None]).float().to(device=device)

        history['ins'].append(np.array(obs)[None])
        history['hx'].append(hx.data.numpy())
        history['cx'].append(cx.data.numpy())
        history['logits'].append(action_dist.logits.data.numpy())
        history['values'].append(value_tensor.data.numpy())
        history['outs'].append(action_dist.probs.data.numpy())
        history['embed'].append(embed_arr[None])
    return history

def run_through_model(policy, history, idx, interp_func, mask=None, mode='actor'):
    if mask is None:
        im = history['ins'][idx]
    else:
        im = interp_func(history['ins'][idx], mask).astype(np.float32) # perturb input
    obs_tensor = torch.from_numpy(im)
    embed_tensor = torch.from_numpy(history['embed'][idx]).float()
    hx = torch.from_numpy(history['hx'][idx])
    cx = torch.from_numpy(history['cx'][idx])
    a_dist, v_tensor, hx, cx = policy(obs_tensor, hx, cx, embed_tensor)
    return a_dist.logits.data.numpy() if mode == 'actor' else v_tensor.data.numpy()

def score_frame(policy, history, idx, radius=5, density=5, interp_func=occlude, mode='actor'):
    """
        @radius: radius of blur
        @density: density of scores (if d==1, then get a score for every pixel...
        if d==2 then every other, which is 25% of total pixels for a 2D image)
    """
    assert mode in ['actor', 'critic'], 'mode must be either "actor" or "critic"'
    L = run_through_model(policy, history, idx, interp_func, mask=None, mode=mode)
    scores = np.zeros((int(84 / density) + 1, int(84 / density) + 1)) # saliency scores S(t,i,j)
    for i in range(0, 84, density):
        for j in range(0, 84, density):
            mask = get_mask(center=[i, j], size=[84, 84, 2], r=radius)
            l = run_through_model(policy, history, idx, interp_func, mask=mask, mode=mode)
            scores[int(i / density), int(j / density)] = 0.5 * np.power((L - l), 2).sum()
    pmax = scores.max()
    scores = imresize(scores, size=[84, 84], interp='bilinear').astype(np.float32)
    return pmax * scores / scores.max()

def saliency_frame(saliency, frame, fudge_factor=100, channel=2, sigma=0):
    """
        sometimes saliency maps are a bit clearer if you blur them
        slightly...sigma adjusts the radius of that blur
    """
    pmax = saliency.max()
    S = saliency if sigma == 0 else gaussian_filter(saliency, sigma=sigma)
    S -= S.min(); S = fudge_factor * pmax * S / S.max()
    S = S[:,:,np.newaxis].astype('uint16')
    I = (frame * 255.).astype('uint16')
    if channel == 0:
        I = np.concatenate((S, I), axis=2)
    else:
        I = np.concatenate((I, S), axis=2)
    return I.clip(1, 255).astype('uint8')

if __name__=='__main__':
    args = parse_args()
    env = gym.make(args.env)
    obs_shape = env.observation_space.shape
    act_dim = env.action_space.n

    # load model
    model = ConvLSTMPolicy(input_size=obs_shape, output_size=act_dim)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))

    # rollout and get saliency maps
    history = rollout(env, model, 'cpu', render=args.render)
    amap_frames = []; cmap_frames = []
    for frame_idx in tqdm(range(len(history['ins']))):
        actor_saliency = score_frame(model, history, frame_idx, mode='actor')
        critic_saliency = score_frame(model, history, frame_idx, mode='critic')

        # display visualization
        frame = history['ins'][frame_idx].squeeze().copy()
        actor_map = saliency_frame(actor_saliency, frame, fudge_factor=100, channel=2) # blue vis; yellow bg
        critic_map = saliency_frame(critic_saliency, frame, fudge_factor=int(3e5), channel=0) # red vis; blueish background

        amap_frames.append(actor_map)
        cmap_frames.append(critic_map)

    imageio.mimsave('base_actor.gif', amap_frames)
    imageio.mimsave('base_critic.gif', cmap_frames)
