from .base_task import BaseTask
from core.data.parameters import PData
from core.data.multitask_parameters import MultitaskPData
from core.data.multitask_episode_parameters import MultitaskEpisodePData
from core.utils.utils import *
from core.utils import *
from core.utils.rl_zoo3 import RLZoo3Trainer
from copy import deepcopy
import hydra
import numpy as np
import torch.nn as nn

import gymnasium as gym
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from rl_zoo3 import train
import time
import dill
from joblib import Parallel, delayed
import multiprocessing

from core.utils.env import make_vector_env, make_gym_env

def evaluate_task(task_name, task_config, param, model, train_layer):
    outputs = {}
    time1 = time.time()
    model = hydra.utils.instantiate(task_config.train.alg, env=env)
    normalization_path = getattr(task_config.param, "normalization_path", None)
    env = make_vector_env(task_config.env.name, task_config.env.num_envs, envpool=False,
                          normalization_path=normalization_path)
    target_num = 0
    for name, module in model.policy.named_parameters():
        if name in train_layer:
            target_num += torch.numel(module)
    params_num = torch.squeeze(param).shape[0]
    assert target_num == params_num

    time2 = time.time()

    partial_reverse_tomodel(param, model.policy, train_layer).to(param.device)

    time3 = time.time()

    return_mean, return_std = evaluate_policy(model, env=env, n_eval_episodes=task_config.env.n_eval_episodes)
    outputs[task_name] = return_mean

    time4 = time.time()
    #print(f"Task {task_name} eval time: {time4 - time1} seconds")

    return outputs


class MultitaskRL(BaseTask):
    def __init__(self, config, **kwargs):
        super(MultitaskRL, self).__init__(config, **kwargs)

    def init_task_data(self):
        task_data = {}
        print(self.cfg.tasks)
        for task_name, task_config in self.cfg.tasks.items():
            print(task_name, task_config)
            normalization_path = getattr(task_config.param, "normalization_path", None)
            env = make_vector_env(task_config.env.name, task_config.env.num_envs, envpool=False, normalization_path=normalization_path)
            task_data[task_name] = env
        return task_data

    def set_param_data(self):
        if True or self.cfg.load_episode_vae: #TODO:修改
            self.pdata = MultitaskEpisodePData(self.cfg)
        else:
            self.pdata = MultitaskPData(self.cfg)
        self.models = {}
        self.train_layers = {}
        for task_name, task_config in self.cfg.tasks.items():
            model = hydra.utils.instantiate(task_config.train.alg, env=self.task_data[task_name])
            model.policy = self.pdata.get_model()[task_name]
            train_layer = self.pdata.get_train_layer()[task_name]
            self.models[task_name] = model
            self.train_layers[task_name] = train_layer
        return self.pdata

    def test_g_model(self, params):
        outputs = {}
        for task_name, task_config in self.cfg.tasks.items():
            if task_name not in params:
                continue
            time1 = time.time()

            param = torch.squeeze(params[task_name])
            model = self.models[task_name]
            train_layer = self.train_layers[task_name]
            env = self.task_data[task_name]

            target_num = 0
            for name, module in model.policy.named_parameters():
                if name in train_layer:
                    target_num += torch.numel(module)
            params_num = torch.squeeze(param).shape[0]  # + 30720
            assert target_num == params_num

            time2 = time.time()

            partial_reverse_tomodel(param, model.policy, train_layer).to(param.device)

            time3 = time.time()

            return_mean, return_std = evaluate_policy(model, env=env, n_eval_episodes=task_config.env.n_eval_episodes)
            outputs[task_name] = return_mean

            time4 = time.time()
            #print("task eval time", time4 - time1)
        return outputs

    # def test_g_model_gg(self, params):
    #     time1 = time.time()
    #     tasks = []
    #     for task_name, task_config in self.cfg.tasks.items():
    #         param = params[task_name]
    #         #model = self.models[task_name]
    #         train_layer = self.train_layers[task_name]
    #         #env = self.task_data[task_name]
    #         tasks.append((task_name, task_config, param, train_layer))
    #
    #     # with multiprocessing.Pool(processes=len(tasks)) as pool:
    #     #     results = pool.starmap(evaluate_task, tasks)
    #     results = Parallel(n_jobs=len(tasks))(delayed(evaluate_task)(*task) for task in tasks)
    #
    #     outputs = {}
    #     for result in results:
    #         print(result)
    #         outputs.update(result)
    #     time2 = time.time()
    #     print("task all eval time", time2 - time1)
    #     return outputs

    def train_for_data(self):
        raise NotImplementedError("Multi task RL cannot be trained simultaneously")

    def train(self, net, criterion, optimizer, trainloader, epoch):
        pass

    def test(self, net, criterion, testloader):
        pass
