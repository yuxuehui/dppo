# Policy Diffusion Network

## Installation

1. Clone the repository

2. Create a new Conda environment and activate it: 

```bash
conda create -n pdn python=3.10 -y
conda activate pdn
pip install -r requirements.txt
```

## training

### Step 1: train RL agents

We use stable-baselines3 and rlzoo to train RL policies:
```bash
python task_training.py task=data/CartPole-v1
python task_training.py task=data/Atari-Assault
python task_training.py task=data/Walker2d-v4
```

### Step 2: train diffusion models

The output is in `outputs/<task_name>` folder. The accuracy in logs means the return of the RL agent. 

train for single task:
```bash
python train.py system=multitask_ae_ddpm task=train/CartPole-v1 device.cuda_visible_devices=0
python train.py system=multitask_ae_ddpm task=train/Atari-Assault device.cuda_visible_devices=0
python train.py system=multitask_ae_ddpm task=train/Walker2d-v4 device.cuda_visible_devices=0

```
train for multitasks:
```bash
python train.py system=multitask_ae_ddpm task=train/mix system.ae_model.condition_num=3 system.model.condition_num=3 device.cuda_visible_devices=0
```
You need to modify condition_num to match the number of tasks. 

The example for multitasks is in configs/train/mix.yaml, you can modify it according to that format. 

train for conditional multitasks_episode_ddpm:
为了泛化任务，将condition用各种形式embedding，已经尝试过的：episode vae，CLIP，VIT,默认使用VIT
暂时将压缩参数的autoencoder去掉了，所以system.model.condition_num=0

```bash
单任务：
python train.py system=multitask_episode_ddpm task=train/Atari-Assault.yaml system.model.condition_num=0 device.cuda_visible_devices=0

多任务：
python train.py system=multitask_episode_ddpm task=train/mix5.yaml system.model.condition_num=0 device.cuda_visible_devices=1

如需观测在未训练的环境上的效果还需修改multitask_episode_ddpm.yaml中的no_train_last参数，例：
python train.py system=multitask_episode_ddpm task=train/8task9234.yaml system.model.condition_num=0 system.no_train_last=True system.validate_only_last_two=True device.cuda_visible_devices=0
即只训练前4个task，最后一个task不训练
```

python train.py system=multitask_episode_ddpm task=train/8task9234.yaml system.model.condition_num=0 system.no_train_last=True system.validate_only_last_two=True system.erase_ratio=0.25 device.cuda_visible_devices=2
python train.py system=multitask_episode_ddpm task=train/8task9234.yaml system.model.condition_num=0 system.no_train_last=True system.validate_only_last_two=True system.erase_ratio=0.5 device.cuda_visible_devices=3
python train.py system=multitask_episode_ddpm task=train/8task9234.yaml system.model.condition_num=0 system.no_train_last=True system.validate_only_last_two=True system.erase_ratio=0.1 device.cuda_visible_devices=4
python train.py system=multitask_episode_ddpm task=train/8task9234.yaml system.model.condition_num=0 system.no_train_last=True system.validate_only_last_two=True system.erase_ratio=0 device.cuda_visible_devices=5

python train.py system=multitask_episode_ddpm task=train/8task9234.yaml system.model.condition_num=0 system.no_train_last=True system.erase_ratio=0 system.train.ddpm_eval_batch_size=50 device.cuda_visible_devices=2
python train.py system=multitask_episode_ddpm task=train/8task9234.yaml system.model.condition_num=0 system.no_train_last=True system.erase_ratio=0.1 system.train.ddpm_eval_batch_size=50 device.cuda_visible_devices=3
python train.py system=multitask_episode_ddpm task=train/8task9234.yaml system.model.condition_num=0 system.no_train_last=True system.erase_ratio=0.25 system.train.ddpm_eval_batch_size=50 device.cuda_visible_devices=4
python train.py system=multitask_episode_ddpm task=train/8task9234.yaml system.model.condition_num=0 system.no_train_last=True system.erase_ratio=0.5 system.train.ddpm_eval_batch_size=50 device.cuda_visible_devices=5
python train.py system=multitask_episode_ddpm task=train/8task9234.yaml system.model.condition_num=0 system.no_train_last=True system.erase_ratio=0.75 system.train.ddpm_eval_batch_size=50 device.cuda_visible_devices=6
python train.py system=multitask_episode_ddpm task=train/8task9234.yaml system.model.condition_num=0 system.no_train_last=True system.erase_ratio=0.9 system.train.ddpm_eval_batch_size=50 device.cuda_visible_devices=7
python train.py system=multitask_episode_ddpm task=train/8task9234.yaml system.model.condition_num=0 system.no_train_last=True system.erase_ratio=1 system.train.ddpm_eval_batch_size=50 device.cuda_visible_devices=7