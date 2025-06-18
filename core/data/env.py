from typing import Optional

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvObs,
    VecEnvStepReturn,
)

import envpool
from envpool.python.protocol import EnvPool


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


def make_envpool_env(env_id, num_envs, seed=None, **kwargs):
    env = envpool.make(env_id, env_type="gym", num_envs=num_envs, seed=seed, **kwargs)
    env.spec.id = env_id
    env = VecAdapter(env)
    env = VecMonitor(env)
    return env
