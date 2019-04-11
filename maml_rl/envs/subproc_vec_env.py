import numpy as np
import multiprocessing as mp
import gym
import sys
is_py2 = (sys.version[0] == '2')
if is_py2:
    import Queue as queue
else:
    import queue as queue
from gym import spaces
import cloudpickle
import pickle

class CloudpickleWrapper(object):
    def __init__(self, var):
        """
        Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)

        :param var: (Any) the variable you wish to wrap for pickling with cloudpickle
        """
        self.var = var

    def __getstate__(self):
        return cloudpickle.dumps(self.var)

    def __setstate__(self, obs):
        self.var = pickle.loads(obs)

def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    global_done = False
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                observation, reward, done, info = env.step(data)
                if done:
                    observation = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == 'reset':
                observation = env.reset()
                remote.send(observation)
            elif cmd == 'render':
                remote.send(env.render(*data[0], **data[1]))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            else:
                raise NotImplementedError
        except EOFError:
            break

def _flatten_obs(obs):
    assert isinstance(obs, (list, tuple))
    assert len(obs) > 0
    if isinstance(obs[0], dict):
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    else:
        return np.stack(obs)

class SubprocVecEnv(gym.Env):
    def __init__(self, env_factory):
        self.waiting = False
        self.closed = False
        n_envs = len(env_factory)
        ctx = mp.get_context('spawn')
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.workers = [ctx.Process(target=_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_factory)]
        for worker in self.workers:
            worker.daemon = True
            worker.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_envs = len(env_factory)

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        assert not self.closed
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        assert not self.closed
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        observations, rewards, dones, infos = zip(*results)
        return _flatten_obs(observations), np.stack(rewards), np.stack(dones), infos

    def reset(self):
        assert not self.closed
        for remote in self.remotes:
            remote.send(('reset', None))
        return _flatten_obs([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for worker in self.workers:
            worker.join()
        