from typing import Optional

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvObs,
    VecEnvStepReturn,
)

import envpool
import gymnasium as gym
from envpool.python.protocol import EnvPool
from stable_baselines3.common.atari_wrappers import AtariWrapper

from stable_baselines3.common.env_util import make_atari_env, make_vec_env
import os
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecFrameStack,
    VecNormalize,
    VecTransposeImage,
    is_vecenv_wrapped,
)


def is_atari_env(env_id):
    return "NoFrameskip" in env_id


def make_vector_env(env_id, num_envs, envpool=False, normalization_path=None, **kwargs):
    if num_envs == 1:
        return make_gym_env(env_id)
    if envpool:
        return make_envpool_env(env_id, num_envs, **kwargs)
    if is_atari_env(env_id):
        #return VecMonitor(SubprocVecEnv([lambda: AtariWrapper(gym.make(env_id)) for _ in range(num_envs)]))
        env = make_atari_env(env_id, n_envs=num_envs)
        env = VecFrameStack(env, n_stack=4)
        return env
    elif normalization_path != None:
        env = make_vec_env(env_id, n_envs=num_envs)
        env = _maybe_normalize(normalization_path, env, eval_env=True)
        return env
    else:
        return VecMonitor(SubprocVecEnv([lambda: gym.make(env_id) for _ in range(num_envs)]))

def make_gym_env(env_id):
    if is_atari_env(env_id):
        env = AtariWrapper(gym.make(env_id))
    else:
        env = gym.make(env_id)
    return env


def convert_gym_space(space):
    if space.__class__.__name__ == "Box":
        new_space = spaces.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
    elif space.__class__.__name__ == "Discrete":
        new_space = spaces.Discrete(n=space.n)
    else:
        raise NotImplementedError(f"Space type {type(space)} is not supported")
    return new_space


class VecAdapter(VecEnvWrapper):
    """
    Convert EnvPool object to a Stable-Baselines3 (SB3) VecEnv.

    :param venv: The envpool object.
    """

    def __init__(self, venv: EnvPool):
        # Retrieve the number of environments from the config
        venv.num_envs = venv.spec.config.num_envs
        super().__init__(venv=venv, observation_space=convert_gym_space(venv.observation_space), action_space=convert_gym_space(venv.action_space))

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def reset(self) -> VecEnvObs:
        return self.venv.reset()[0]

    def seed(self, seed: Optional[int] = None) -> None:
        # You can only seed EnvPool env by calling envpool.make()
        pass

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, terms, truncs, info_dict = self.venv.step(self.actions)
        dones = terms + truncs
        infos = []
        # Convert dict to list of dict
        # and add terminal observation
        for i in range(self.num_envs):
            infos.append({key: info_dict[key][i] for key in info_dict.keys() if isinstance(info_dict[key], np.ndarray)})
            if dones[i]:
                infos[i]["terminal_observation"] = obs[i]
                obs[i] = self.venv.reset(np.array([i]))[0]
        return obs, rewards, dones, infos


class EnvPoolAtariWrapper(VecAdapter):
    def __init__(self, env):
        super().__init__(venv=env)

    def reset(self):
        obs = super().reset()
        return obs[:, -1:]

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = super().step_wait()
        obs = obs[:, -1:]
        return obs, rewards, dones, infos


def envpool_env_id(env_id):
    if "NoFrameskip" in env_id:
        return env_id.split("NoFrameskip")[0] + "-v5"
    return env_id


def make_envpool_env(env_id, num_envs, seed=None, **kwargs):
    is_atari = is_atari_env(env_id)
    env_id = envpool_env_id(env_id)
    if seed:
        env = envpool.make(env_id, env_type="gym", num_envs=num_envs, seed=seed, **kwargs)
    else:
        env = envpool.make(env_id, env_type="gym", num_envs=num_envs, **kwargs)
    env.spec.id = env_id
    if is_atari:
        env = EnvPoolAtariWrapper(env)
    else:
        env = VecAdapter(env)
    env = VecMonitor(env)
    return env

def _maybe_normalize(normalization_path, env: VecEnv, eval_env: bool) -> VecEnv:
    """
    Wrap the env into a VecNormalize wrapper if needed
    and load saved statistics when present.

    :param env:
    :param eval_env:
    :return:
    """

    if os.path.exists(normalization_path):
        print("Loading saved VecNormalize stats")
        env = VecNormalize.load(normalization_path, env)
        # Deactivate training and reward normalization
        if eval_env:
            env.training = False
            env.norm_reward = False
    return env
