import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate
from torch.utils.data import Dataset
from .base import DataBase
import os
import pickle


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
        


class MultitaskPData(DataBase):
    def __init__(self, cfg, model=None, **kwargs):
        super(MultitaskPData, self).__init__(cfg, **kwargs)
        
        self.task_configs = self.cfg.tasks
        self.task_infos = {}
        for task_name, task_config in self.task_configs.items():
            task_config = task_config.param
            data_root = getattr(task_config, "data_root", "./data")
            data_size = getattr(task_config, "k", 200)

            state = torch.load(data_root, map_location="cpu")

            if "model" in state:
                model = state["model"]
                model.eval()
                model.to("cpu")
                model.requires_grad_(False)

            pdata = state["pdata"]
            performance = state["performance"]
            train_layer = state["train_layer"]
            
            task_info = dict(model=model, pdata=pdata, performance=performance, train_layer=train_layer, data_size=data_size)
            self.task_infos[task_name] = task_info
            print(f"Task {task_name} metric: max {np.max(performance)}, mean {np.mean(performance)}, median {np.median(performance)}")
        
        self.data_size = min([task_info["data_size"] for task_info in self.task_infos.values()])
        self.batch_size = min([getattr(task_config, "batch_size", self.data_size) for task_config in self.task_configs.values()])
        self.num_workers = min([getattr(task_config, "num_workers", 1) for task_config in self.task_configs.values()])

    def get_train_layer(self):
        return {task_name: task_info["train_layer"] for task_name, task_info in self.task_infos.items()}

    def get_model(self):
        return {task_name: task_info["model"] for task_name, task_info in self.task_infos.items()}

    @property
    def train_dataset(self):
        return ConcatDataset(*[Parameters(task_info["pdata"], self.data_size) for task_info in self.task_infos.values()])

    @property
    def val_dataset(self):
        return ConcatDataset(*[Parameters(task_info["pdata"], self.data_size) for task_info in self.task_infos.values()])

    @property
    def test_dataset(self):
        return ConcatDataset(*[Parameters(task_info["pdata"], self.data_size) for task_info in self.task_infos.values()])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, collate_fn=ConcatCollator(), batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, collate_fn=ConcatCollator(), batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, collate_fn=ConcatCollator(), batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, persistent_workers=True)


class Parameters(Dataset):
    def __init__(self, batch, k):
        if len(batch) != k:
            indices = range(0, len(batch), int(len(batch)/k))
            self.data = [batch[i] for i in indices]
        else:
            self.data = batch[:k]

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self) -> int:
        return len(self.data)
