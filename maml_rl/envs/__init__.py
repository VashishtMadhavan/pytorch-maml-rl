from gym.envs.registration import register
# from .coinrun.coinrunenv import init_args_and_threads, CoinRunVecEnv
# from .subproc_vec_env import VecFrameStack

# def CoinRunEnv(game_type='standard', num_envs=4, frame_stack=1):
#     init_args_and_threads()
#     env = CoinRunVecEnv(game_type=game_type, num_envs=num_envs)
#     env = VecFrameStack(env, frame_stack)
#     return env

# Bandit
# ----------------------------------------

for k in [5, 10, 50]:
    for n in [10, 100, 500]:
        register(
            'Bandit-K{0}-N{1}-v0'.format(k, n),
            entry_point='maml_rl.envs.bandit:BernoulliBanditEnv',
            kwargs={'k': k, 'n': n},
            max_episode_steps=1000
        )

# TabularMDP
# ----------------------------------------

register(
    'TabularMDP-v0',
    entry_point='maml_rl.envs.mdp:TabularMDPEnv',
    kwargs={'num_states': 10, 'num_actions': 5},
    max_episode_steps=10
)

# Pygames
# ----------------------------------------

register(
    'CustomGame-v0',
    entry_point='maml_rl.envs.utils:universe_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.game:CustomGameEnv'},
    max_episode_steps=100
)

register(
    'CustomGame-v1',
    entry_point='maml_rl.envs.utils:universe_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.meta_game:CustomGameMetaEnv'},
    max_episode_steps=200
)

register(
    'OriginalGame-v0',
    entry_point='maml_rl.envs.utils:universe_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.orig_game:OriginalGameEnv'},
    max_episode_steps=100
)

# GridGame
# ----------------------------------------
register(
    'GridGameTrain-v0',
    entry_point='maml_rl.envs.grid_game:GridGameEnv',
    kwargs={'setup': 0, 'task': {}},
    max_episode_steps=25
)

register(
    'GridGameSmallTest-v0',
    entry_point='maml_rl.envs.grid_game:GridGameEnv',
    kwargs={'setup': 1, 'task': {}},
    max_episode_steps=25
)

register(
    'GridGameTest-v0',
    entry_point='maml_rl.envs.grid_game:GridGameEnv',
    kwargs={'setup': 2, 'task': {}},
    max_episode_steps=25
)


# Mujoco
# ----------------------------------------

register(
    'AntVel-v1',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.ant:AntVelEnv'},
    max_episode_steps=200
)

register(
    'AntDir-v1',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.ant:AntDirEnv'},
    max_episode_steps=200
)

register(
    'AntPos-v0',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.ant:AntPosEnv'},
    max_episode_steps=200
)

register(
    'HalfCheetahVel-v1',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.half_cheetah:HalfCheetahVelEnv'},
    max_episode_steps=200
)

register(
    'HalfCheetahDir-v1',
    entry_point='maml_rl.envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.half_cheetah:HalfCheetahDirEnv'},
    max_episode_steps=200
)

# 2D Navigation
# ----------------------------------------

register(
    '2DNavigation-v0',
    entry_point='maml_rl.envs.navigation:Navigation2DEnv',
    max_episode_steps=100
)
