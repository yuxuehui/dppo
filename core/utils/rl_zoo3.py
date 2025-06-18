import argparse
import difflib
import importlib
import os
import time
import uuid

import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
import torch as th
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.base_class import BaseAlgorithm

# Register custom envs
import rl_zoo3.import_envs  # noqa: F401
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.utils import ALGOS, StoreDict
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecFrameStack,
    VecNormalize,
    VecTransposeImage,
    is_vecenv_wrapped,
)



class RLZoo3Trainer():
    def __init__(self, algo, env):
        self.algo = algo
        self.env = env

    def setup_trainer(self, log_folder="logs", **kwargs):
        args_dict = {
            "algo": self.algo,
            "env": self.env,
            "tensorboard_log": log_folder,
            "trained_agent": "",
            "truncate_last_trajectory": True,
            "n_timesteps": -1,
            "num_threads": -1,
            "log_interval": 100,
            "eval_freq": 25000,
            "optimization_log_path": None,
            "eval_episodes": 5,
            "n_eval_envs": 1,
            "save_freq": -1,
            "save_replay_buffer": False,
            "log_folder": log_folder,
            "seed": -1,
            "vec_env": "dummy",
            "device": "cuda",
            "n_trials": 500,
            "max_total_trials": None,
            "optimize_hyperparameters": False,
            "no_optim_plots": False,
            "n_jobs": 1,
            "sampler": "tpe",
            "pruner": "median",
            "n_startup_trials": 10,
            "n_evaluations": None,
            "storage": None,
            "study_name": None,
            "verbose": 1,
            "gym_packages": [],
            "env_kwargs": {},
            "eval_env_kwargs": {},
            "hyperparams": {},
            "conf_file": None,
            "uuid": False,
            "track": False,
            "wandb_project_name": "sb3",
            "wandb_entity": None,
            "progress": False,
            "wandb_tags": []
        }
        args_dict.update(kwargs)
        args = argparse.Namespace(**args_dict)

        # Going through custom gym packages to let them register in the global registry
        for env_module in args.gym_packages:
            importlib.import_module(env_module)

        env_id = args.env
        registered_envs = set(gym.envs.registry.keys())

        # If the environment is not found, suggest the closest match
        if env_id not in registered_envs:
            try:
                closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
            except IndexError:
                closest_match = "'no close match found...'"
            raise ValueError(f"{env_id} not found in gym registry, you maybe meant {closest_match}?")

        # Unique id to ensure there is no race condition for the folder creation
        uuid_str = f"_{uuid.uuid4()}" if args.uuid else ""
        if args.seed < 0:
            # Seed but with a random one
            args.seed = np.random.randint(2**32 - 1, dtype="int64").item()  # type: ignore[attr-defined]

        set_random_seed(args.seed)

        # Setting num threads to 1 makes things run faster on cpu
        if args.num_threads > 0:
            if args.verbose > 1:
                print(f"Setting torch.num_threads to {args.num_threads}")
            th.set_num_threads(args.num_threads)

        if args.trained_agent != "":
            assert args.trained_agent.endswith(".zip") and os.path.isfile(
                args.trained_agent
            ), "The trained_agent must be a valid path to a .zip file"

        print("=" * 10, env_id, "=" * 10)
        print(f"Seed: {args.seed}")

        run_name = f"{args.env}__{args.algo}__{args.seed}__{int(time.time())}"
        args.tensorboard_log = f"{args.log_folder}/runs/{run_name}"

        exp_manager = ExperimentManager(
            args,
            args.algo,
            env_id,
            args.log_folder,
            args.tensorboard_log,
            args.n_timesteps,
            args.eval_freq,
            args.eval_episodes,
            args.save_freq,
            args.hyperparams,
            args.env_kwargs,
            args.eval_env_kwargs,
            args.trained_agent,
            args.optimize_hyperparameters,
            args.storage,
            args.study_name,
            args.n_trials,
            args.max_total_trials,
            args.n_jobs,
            args.sampler,
            args.pruner,
            args.optimization_log_path,
            n_startup_trials=args.n_startup_trials,
            n_evaluations=args.n_evaluations,
            truncate_last_trajectory=args.truncate_last_trajectory,
            uuid_str=uuid_str,
            seed=args.seed,
            log_interval=args.log_interval,
            save_replay_buffer=args.save_replay_buffer,
            verbose=args.verbose,
            vec_env_type=args.vec_env,
            n_eval_envs=args.n_eval_envs,
            no_optim_plots=args.no_optim_plots,
            device=args.device,
            config=args.conf_file,
            show_progress=args.progress,
        )
        #exp_manager._preprocess_normalization = lambda x: {k: v for k, v in x.items() if k != "normalize"}

        # Prepare experiment and launch hyperparameter optimization if needed
        model, saved_hyperparams = exp_manager.setup_experiment()
        self.saved_hyperparams = saved_hyperparams
        self.exp_manager = exp_manager
        self.model = model
        return model
    
    def train(self):
        kwargs = {}
        if self.exp_manager.log_interval > -1:
            kwargs = {"log_interval": self.exp_manager.log_interval}

        if len(self.exp_manager.callbacks) > 0:
            kwargs["callback"] = self.exp_manager.callbacks

        self.model.learn(self.exp_manager.n_timesteps, **kwargs)

    def get_model(self, **kwargs):
        exp_manager = ExperimentManager(
            argparse.Namespace(),
            self.algo,
            self.env,
            "/tmp",
        )
        model, saved_hyperparams = exp_manager.setup_experiment()
        return model

    def _maybe_normalize(self, save_path, env: VecEnv, eval_env: bool) -> VecEnv:
        """
        Wrap the env into a VecNormalize wrapper if needed
        and load saved statistics when present.

        :param env:
        :param eval_env:
        :return:
        """
        # Pretrained model, load normalization
        path_ = os.path.join(save_path, "vecnormalize.pkl")

        if os.path.exists(path_):
            print("Loading saved VecNormalize stats")
            env = VecNormalize.load(path_, env)
            # Deactivate training and reward normalization
            if eval_env:
                env.training = False
                env.norm_reward = False
        return env

    def save_normalize(self, save_path) -> None:
        """
        Save trained model optionally with its replay buffer
        and ``VecNormalize`` statistics

        :param model:
        """
        print(f"Saving to {save_path}")
        #model.save(f"{save_path}/{self.env_name}")
        if self.exp_manager.normalize:
            # Important: save the running average, for testing the agent we need that normalization
            vec_normalize = self.model.get_vec_normalize_env()
            assert vec_normalize is not None
            vec_normalize.save(os.path.join(save_path, "vecnormalize.pkl"))

