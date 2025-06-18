import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate
from torch.utils.data import Dataset
from .base import DataBase
import os
import pickle
import random
from PIL import Image


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class ConcatCollator():
    def __init__(self):
        pass

    def __call__(self, batch):
        batch_size = len(batch)
        concat_len = len(batch[0])
        return tuple(default_collate([batch[j][i] for j in range(batch_size)]) for i in range(concat_len))
        


class MultitaskEpisodePData(DataBase):
    def __init__(self, cfg, model=None, **kwargs):
        super(MultitaskEpisodePData, self).__init__(cfg, **kwargs)
        
        self.task_configs = self.cfg.tasks
        self.task_infos = {}

        for task_name, task_config in self.task_configs.items():
            task_config = task_config.param
            data_root = getattr(task_config, "data_root", "./data")
            data_size = getattr(task_config, "k", 200)
            episodes_data_path = os.path.join(os.path.dirname(data_root), 'episodes_data2.pkl')

            state = torch.load(data_root, map_location="cpu")
            #episodes_data = []
            with open(episodes_data_path, 'rb') as f:
                episodes_data = pickle.load(f)

            # 将数据展平为 [200, 3, 30, (obs)]
            filtered_episodes_data = [[episode for episode in episodes if len(episode) >= self.cfg.episode_len] for episodes in episodes_data]


            if "model" in state:
                model = state["model"]
                model.eval()
                model.to("cpu")
                model.requires_grad_(False)

            pdata = state["pdata"]
            performance = state["performance"]
            train_layer = state["train_layer"]
            
            task_info = dict(model=model, pdata=pdata, episodes_data=filtered_episodes_data, performance=performance, train_layer=train_layer, data_size=data_size)
            self.task_infos[task_name] = task_info
            print(f"Task {task_name} metric: max {np.max(performance)}, mean {np.mean(performance)}, median {np.median(performance)}")
        
        self.data_size = min([task_info["data_size"] for task_info in self.task_infos.values()])
        #self.batch_size = min([getattr(task_config, "batch_size", self.data_size) for task_config in self.task_configs.values()])
        self.batch_size = min(
            [getattr(task_config, "batch_size", int(self.data_size)) for task_config in self.task_configs.values()])
        self.num_workers = min([getattr(task_config, "num_workers", 1) for task_config in self.task_configs.values()])

    def get_train_layer(self):
        return {task_name: task_info["train_layer"] for task_name, task_info in self.task_infos.items()}

    def get_model(self):
        return {task_name: task_info["model"] for task_name, task_info in self.task_infos.items()}

    @property
    def train_dataset(self):
        return ConcatDataset(*[Parameters(task_info["pdata"], task_info["episodes_data"], self.data_size, self.cfg.episode_len) for task_info in self.task_infos.values()])

    @property
    def val_dataset(self):
        return ConcatDataset(*[Parameters(task_info["pdata"], task_info["episodes_data"], self.data_size, self.cfg.episode_len) for task_info in self.task_infos.values()])

    @property
    def test_dataset(self):
        return ConcatDataset(*[Parameters(task_info["pdata"], task_info["episodes_data"], self.data_size, self.cfg.episode_len) for task_info in self.task_infos.values()])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, collate_fn=ConcatCollator(), batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, collate_fn=ConcatCollator(), batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, collate_fn=ConcatCollator(), batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, persistent_workers=True)


class Parameters(Dataset):
    def __init__(self, pdata, episodes_data, k, episode_len):
        self.pdata = pdata
        self.episodes_data = episodes_data
        self.k = k
        self.episode_len = episode_len
        if len(pdata) != k:
            indices = range(0, len(pdata), int(len(pdata) / k))
            self.pdata = [pdata[i] for i in indices]
        else:
            self.pdata = pdata[:k]

    def __getitem__(self, item):
        pdata_item = self.pdata[item]
        episodes_data_item = self.episodes_data[item]
        random_episode = random.choice(episodes_data_item)
        start_idx = random.randint(0, len(random_episode) - 10)  # 40 是 50 - 10
        sub_episode = random_episode[start_idx:start_idx + 10]
        states = np.array([step for step in sub_episode])
        states = torch.tensor(states, dtype=torch.float32)
        episode = states / 255.0
        return pdata_item, episode

    # def __getitem__(self, item):
    #     pdata_item = self.pdata[item]
    #     episodes_data_item = self.episodes_data[item]
    #     random_episode = random.choice(episodes_data_item)
    #     start_idx = random.randint(0, len(random_episode) - self.episode_len)  # 40 是 50 - 10
    #     sub_episode = random_episode[start_idx:start_idx + self.episode_len]
    #     states = np.array([step[:, :, 0] for step in sub_episode])
    #     #states = np.array([step for step in sub_episode])
    #     #episode = [Image.fromarray(frame) for frame in states]
    #
    #     #print("episode.shape: ", episode.shape)
    #     #episode = torch.tensor(states, dtype=torch.float32)
    #     episode = states
    #     #episode = states / 255.0
    #     return pdata_item, episode

    def __len__(self) -> int:
        return len(self.pdata)
