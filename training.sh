#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# source ~/anaconda3/bin/activate pdn

# train on atari
nohup  python train.py system=multitask_episode_ddpm task=train/mix5.yaml system.sampling_method="ddim" system.model.condition_num=0 device.cuda_visible_devices=0  > ./logs/atari/atari_mix_ddim.log 2>&1 &
nohup  python train.py system=multitask_episode_ddpm task=train/mix5.yaml system.sampling_method="p_sample" system.model.condition_num=0 device.cuda_visible_devices=0  > ./logs/atari/atari_mix_p_sample.log 2>&1 &

# python train.py system=multitask_episode_ddpm task=train/mix5.yaml system.model.condition_num=0 device.cuda_visible_devices=0
