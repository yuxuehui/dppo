from .base_task import BaseTask
from core.data.parameters import PData
from core.utils.utils import *
from core.utils import *
from core.utils.rl_zoo3 import RLZoo3Trainer
from copy import deepcopy
import hydra
import numpy as np
import torch.nn as nn

from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.evaluation import evaluate_policy
from rl_zoo3 import train
from tqdm import tqdm

from core.utils.env import make_vector_env, make_gym_env
from stable_baselines3 import PPO
from tqdm import tqdm
import pickle
import time


class RLTask(BaseTask):
    def __init__(self, config, **kwargs):

        super(RLTask, self).__init__(config, **kwargs)

    def init_task_data(self):
        return make_vector_env(self.cfg.env.name, self.cfg.env.num_envs, normalization_path=os.path.join(self.cfg.train.save_path, "vecnormalize.pkl"))

    # override the abstract method in base_task.py
    def set_param_data(self):
        self.model = hydra.utils.instantiate(self.cfg.train.alg, env=self.task_data)
        self.pdata = PData(self.cfg.param)
        self.model.policy = self.pdata.get_model()
        self.train_layer = self.pdata.get_train_layer()
        return self.pdata

    def test_g_model(self, param):
        target_num = 0
        for name, module in self.model.policy.named_parameters():
            if name in self.train_layer:
                target_num += torch.numel(module)
        params_num = torch.squeeze(param).shape[0]  # + 30720
        assert target_num == params_num
        param = self.pdata.recover_params(torch.squeeze(param))
        partial_reverse_tomodel(param, self.model.policy, self.train_layer).to(param.device)
        return_mean, return_std = evaluate_policy(self.model, env=self.task_data, n_eval_episodes=self.cfg.env.n_eval_episodes)
        output_list = []
        return return_mean, return_std, output_list

    def train_for_data(self):
        config = self.cfg

        # Init environment
        test_env = self.task_data
        
        # Init policy
        save_path = config.train.save_path
        os.makedirs(save_path, exist_ok=True)
        print(f"Save results to {save_path}")

        trainer = RLZoo3Trainer(algo=config.train.algo, env=config.env.name)
        model = trainer.setup_trainer(log_folder=save_path)

        # # Evaluate with random initialization
        # return_mean, return_std = evaluate_policy(model, test_env, n_eval_episodes=config.env.n_eval_episodes)
        # env_name = self.cfg.env.name
        # print("!!!!!----------------------------------------------------!!!!!!")
        # print("!!!!!----------------------------------------------------!!!!!!")
        # print(f"For Task {env_name}:")
        # print(
        #     f"Random initialization performance: mean {return_mean:.2f}, std {return_std:.2f}")
        # print("!!!!!----------------------------------------------------!!!!!!")
        # print("!!!!!----------------------------------------------------!!!!!!")

        train_layers = config.train.train_layers
        for k, p in model.policy.named_parameters():
            print(k, p.shape)
        param_num = sum(p.numel() for k, p in model.policy.named_parameters() if k in train_layers)
        print(f"\nNumber of parameters in the policy: {param_num}\n")

        trainer.train()
        trainer.save_normalize(save_path)


        # Start fine-tuning and save parameters
        fix_partial_model(train_layers, model.policy)
        model.verbose = 0
        saved_models = []
        for group in model.policy.optimizer.param_groups:
            group["lr"] = config.train.finetune_lr

        for i in tqdm(range(config.train.save_model_num)):
            saved_models.append(flat_model(model.policy.state_dict(), train_layers))
            model.learn(config.train.eval_freq, reset_num_timesteps=False)

        #test_env = self.init_task_data()
        test_env = self.task_data
        performances = []
        std = []
        for i, param in enumerate(saved_models):
            model.policy = partial_reverse_tomodel(param, model.policy, train_layers)
            return_mean, return_std = evaluate_policy(model, test_env, n_eval_episodes=config.env.n_eval_episodes)
            performances.append(return_mean)
            std.append(return_std)
            print(f"[{i+1}/{config.train.save_model_num}] Return mean {return_mean:.2f}, std {return_std:.2f}")

        data = {}
        data["cfg"] = OmegaConf.to_container(config, resolve=True)
        data["train_layer"] = train_layers
        data["model"] = model.policy
        data["pdata"] = torch.stack(saved_models, dim=0)
        data["performance"] = performances
        data["std"] = std
        torch.save(data, f"{save_path}/data.pt")
        print(f"Save data to {save_path}/data.pt")

    def train_for_generated_parameter(self):
        config = self.cfg

        # Init environment
        test_env = self.task_data

        # Init policy
        save_path = config.train.save_path
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path+"/generated_random", exist_ok=True)
        print(f"Save results to {save_path}")

        trainer = RLZoo3Trainer(algo=config.train.algo, env=config.env.name, )
        model = trainer.setup_trainer(log_folder=save_path+"/generated_random")

        train_layers = config.train.train_layers
        data_root = save_path + "/data.pt"
        state = torch.load(data_root, map_location="cpu")
        model.policy = state["model"]
        model.policy.to("cuda")
        param_root = save_path + "/generate_param.pt"
        param = torch.squeeze(torch.load(param_root, map_location="cpu"))

        new_param = torch.randn(param.shape)
        param = new_param

        model.policy = partial_reverse_tomodel(param, model.policy, train_layers)
        model.policy.to("cuda")
        #model.policy.optimizer.cuda()
        #print(model)
        #print(model.policy)
        # print(model.policy.optimizer)
        # for param1 in model.policy.parameters():
        #     print(param1.device)
        for state in model.policy.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    #print(v.device)
                    state[k] = v.to("cuda")

        # Evaluate with random initialization
        return_mean, return_std = evaluate_policy(model, test_env, n_eval_episodes=config.env.n_eval_episodes)
        env_name = self.cfg.env.name
        print("!!!!!----------------------------------------------------!!!!!!")
        print("!!!!!----------------------------------------------------!!!!!!")
        print(f"For Task {env_name}:")
        print(
            f"Random initialization performance: mean {return_mean:.2f}, std {return_std:.2f}")
        print("!!!!!----------------------------------------------------!!!!!!")
        print("!!!!!----------------------------------------------------!!!!!!")

        #trainer.train()

    def train(self, net, criterion, optimizer, trainloader, epoch):
        pass

    def test(self, net, criterion, testloader):
        pass

    def generate_episode_data(self, num_episodes=3):
        env = self.task_data
        self.set_param_data()

        all_episodes_data = []

        for _, param in enumerate(tqdm(self.pdata.pdata)):
            target_num = 0
            for name, module in self.model.policy.named_parameters():
                if name in self.train_layer:
                    target_num += torch.numel(module)
            params_num = torch.squeeze(param).shape[0]
            assert target_num == params_num
            param = self.pdata.recover_params(torch.squeeze(param))
            partial_reverse_tomodel(param, self.model.policy, self.train_layer).to(param.device)

            episodes_data = []

            for i in range(num_episodes):
                obs = env.reset()
                dones = [False] * env.num_envs  # 初始化每个环境的done状态
                episode = [[] for _ in range(env.num_envs)]  # 为每个环境初始化一个episode列表

                cur_episode_num = 0
                while not all(dones) and cur_episode_num < 30:  # 只要有一个环境没有完成，就继续
                    actions, _states = self.model.predict(obs)
                    new_obs, rewards, new_dones, infos = env.step(actions)
                    #print(cur_episode_num)
                    for j in range(env.num_envs):
                        if not dones[j]:  # 只记录未完成的环境
                            #episode[j].append((obs[j], actions[j], rewards[j], new_obs[j]))
                            episode[j].append(obs[j])
                    obs = new_obs
                    dones = [done or new_done for done, new_done in zip(dones, new_dones)]
                    cur_episode_num += 1
                #print(i, episode[0])

                episodes_data.extend(episode)  # 将所有环境的episode数据添加到总列表中

            all_episodes_data.append(episodes_data)

        start_time = time.time()
        with open(self.cfg.train.save_path+'/episodes_data2.pkl', 'wb') as f:
            pickle.dump(all_episodes_data, f)
        end_time = time.time()
        print(f"Pickle dump took {end_time - start_time:.2f} seconds")

        env.close()
        return
