import numpy as np
import multiprocessing as mp
import gym
import sys
is_py2 = (sys.version[0] == '2')
if is_py2:
    import Queue as queue
else:
    import queue as queue

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

def _worker(remote, parent_remote, env_fn_wrapper, queue, lock):
    parent_remote.close()
    env = env_fn_wrapper.var()
    task_id = None
    global_done = False

    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                if global_done:
                    observation, reward, done, info = np.zeros(env.observation_space.shape, dtype=np.float32), 0.0, True, {}
                else:
                    observation, reward, done, info = env.step(data)
                if done and (not global_done):
                    with lock:
                        try:
                            task_id = queue.get(True)
                            global_done = (task_id is None)
                        except queue.Empty:
                            global_done = True
                    observation = (np.zeros(env.observation_space.shape, dtype=np.float32) if global_done else env.reset())
                remote.send((observation, reward, done, task_id, info))
            elif cmd == 'reset':
                with lock:
                    try:
                        task_id = queue.get(True)
                        global_done = (task_id is None)
                    except queue.Empty:
                        global_done = True
                    observation = (np.zeros(env.observation_space.shape, dtype=np.float32) if global_done else env.reset())
                remote.send((observation, task_id))
            elif cmd == 'reset_task':
                env.unwrapped.reset_task(data)
                remote.send(True)
            elif cmd == 'get_task':
                remote.send(env.unwrapped.get_task())
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


class SubprocVecEnv(gym.Env):
    def __init__(self, env_factory, queue):
        self.lock = mp.Lock()
        n_envs = len(env_factory)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(n_envs)])
        self.workers = [mp.Process(target=_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn), queue, self.lock))
                          for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_factory)]
        for worker in self.workers:
            worker.daemon = True
            worker.start()
        for remote in self.work_remotes:
            remote.close()
        self.waiting = False
        self.closed = False

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.observation_space = observation_space
        self.action_space = action_space

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        observations, rewards, dones, task_ids, infos = zip(*results)
        return np.stack(observations), np.stack(rewards), np.stack(dones), task_ids, infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        observations, task_ids = zip(*results)
        return np.stack(observations), task_ids

    def reset_task(self, tasks):
        for remote, task in zip(self.remotes, tasks):
            remote.send(('reset_task', task))
        return np.stack([remote.recv() for remote in self.remotes])

    def get_task(self):
        for remote in self.remotes:
            remote.send(('get_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for worker in self.workers:
            worker.join()
        self.closed = True
