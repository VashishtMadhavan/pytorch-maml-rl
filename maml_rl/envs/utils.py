from gym.envs.registration import load
from gym.wrappers.atari_wrappers import WarpFrame, ScaledFloatFrame, MaxAndSkipEnv, ClipRewardEnv, FrameStack
from .normalized_env import NormalizedActionWrapper

def mujoco_wrapper(entry_point, **kwargs):
    # Load the environment from its entry point
    env_cls = load(entry_point)
    env = env_cls(**kwargs)
    # Normalization wrapper
    env = NormalizedActionWrapper(env)
    return env


def universe_wrapper(entry_point, **kwargs):
	# Load the environment from its entry point
    env_cls = load(entry_point)
    env = env_cls(**kwargs)

    # Preprocessing wrappers
    env = MaxAndSkipEnv(env, skip=4)
    env = WarpFrame(env)
    env = ScaledFloatFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 2)
    return env