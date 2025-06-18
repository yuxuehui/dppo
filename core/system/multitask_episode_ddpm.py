import pdb

import hydra.utils
import pytorch_lightning as pl
import torch
from typing import Any
import numpy as np
import torch.nn as nn
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import time
import os
import math
import random

from .base import BaseSystem
from core.utils.ddpm import *
from core.utils.utils import *
from core.module.prelayer.latent_transformer import Param2Latent
from .ddpm import DDPM
import json
from core.episode.episode_vae import EpisodeVAE
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageDraw

class MultitaskEpisode_DDPM(DDPM):
    def __init__(self, config, **kwargs):
        self.task_cfg = config.task
        print(self.task_cfg)

        #ae_model = hydra.utils.instantiate(config.system.ae_model)
        super(MultitaskEpisode_DDPM, self).__init__(config)
        self.save_hyperparameters()
        self.split_epoch = self.train_cfg.split_epoch
        self.ae_use_condition = self.train_cfg.ae_use_condition
        self.loss_func = nn.MSELoss()
        #self.ae_model = ae_model
        self.load_best_ae = False

        self.load_episode_vae = getattr(config.system, "load_episode_vae", None)
        self.no_train_last = getattr(config.system, "no_train_last", None)
        self.validate_only_last_two = getattr(config.system, "validate_only_last_two", None)
        self.load_clip = getattr(config.system, "load_clip", None)
        self.episode_len = getattr(config.system, "episode_len", 30)
        self.num_gpus = torch.cuda.device_count()
        self.testing_erase = getattr(config.system, "testing_erase", False)
        self.erase_ratio = getattr(config.system, "erase_ratio", 0.25)
        if self.load_episode_vae:
            img_input_dim = 84 * 84 * 4  # Atari图像大小
            token_dim = 32
            hidden_dim = 256
            latent_dim = 64
            num_heads = 4
            num_layers = 2
            img_output_dim = 84 * 84 * 4  # Atari图像大小
            patch_size = 84
            episode_vae_path = os.path.join(self.config.project_root, "episode_outputs/5-1/vae-epoch=313-val_loss=0.02.ckpt")
            self.episode_model = EpisodeVAE(img_input_dim, token_dim, hidden_dim, latent_dim, num_heads, num_layers, img_output_dim, patch_size=patch_size, only_ae=True)
            checkpoint = torch.load(episode_vae_path)
            self.episode_model.load_state_dict(checkpoint['state_dict'])
            self.episode_model.eval()

            self.episode_model.cuda()
        if self.load_clip:
            model_name = "/mnt/kaiwu-group-x3/jiateliu/models/clip-vit-base-patch32"
            self.model.cuda()
            self.clip_model = CLIPModel.from_pretrained(model_name)
            self.clip_model.cuda()
            self.clip_model.eval()
            self.processor = CLIPProcessor.from_pretrained(model_name)
         
        self.load_vit = getattr(config.system, "load_vit", None)
        if self.load_vit:
            # Use official Hugging Face model ID with backup offline path
            vit_model_name = "vit-base-patch16-224"  # Local model name
            vit_model_id = "google/vit-base-patch16-224"  # Hugging Face model ID
            
            from transformers import ViTImageProcessor, ViTModel
            try:
                # Try loading from local path first
                print("Attempting to load ViT model from local cache...")
                self.vit_model = ViTModel.from_pretrained(vit_model_name, local_files_only=True)
                self.vit_processor = ViTImageProcessor.from_pretrained(vit_model_name, local_files_only=True)
            except Exception as local_err:
                print(f"Local load failed: {local_err}")
                try:
                    # Try downloading from Hugging Face
                    print(f"Downloading ViT model from Hugging Face: {vit_model_id}")
                    self.vit_model = ViTModel.from_pretrained(vit_model_id, force_download=True)
                    self.vit_processor = ViTImageProcessor.from_pretrained(vit_model_id, force_download=True)
                    
                    # Save for future use
                    print("Saving model locally...")
                    self.vit_model.save_pretrained(vit_model_name)
                    self.vit_processor.save_pretrained(vit_model_name)
                except Exception as e:
                    print(f"Error downloading ViT model: {e}")
                    raise

            self.vit_model.cuda()
            self.vit_model.eval()
            
            # Add dimension reduction layer
            self.vit_feature_reducer = nn.Linear(768, self.model.hidden_size)
            self.vit_feature_reducer.cuda()

        self.load_history_ae = getattr(config.system, "ae_path", None)
        if getattr(config.system, "ae_path", None) is not None:
            checkpoint = torch.load(config.system.ae_path)
            new_state_dict = {}
            for key in checkpoint["state_dict"]:
                if "ae_model." in key:
                    new_key = key.replace("ae_model.", "")
                    new_state_dict[new_key] = checkpoint["state_dict"][key]
            ae_model.load_state_dict(new_state_dict)

        self.all_performance = {}
        for task_name, task_config in self.task_cfg.tasks.items():
            print(task_config)
            task_config = task_config.param
            data_root = getattr(task_config, "data_root", "./data")
            data_size = getattr(task_config, "k", 200)
            state = torch.load(data_root, map_location="cpu")
            performance = state["performance"][:data_size]
            self.all_performance[task_name] = {}
            self.all_performance[task_name]["mean"] = np.mean(performance)
            self.all_performance[task_name]["std"] = np.std(performance)
            self.all_performance[task_name]["max"] = np.max(performance)
            self.all_performance[task_name]["median"] = np.median(performance)

        print("all_preformance: ", self.all_performance)


    def ae_forward(self, batch, condition=None, condition2=None,**kwargs):
        output = self.ae_model(batch, condition=condition, condition2=condition2)
        loss = self.loss_func(batch, output, **kwargs)
        return loss


    def training_step(self, batch, batch_idx, **kwargs):
        ddpm_optimizer = self.optimizers()
        # if self.current_epoch < self.split_epoch:
        #     loss = torch.tensor(0.0).cuda()
        #     ae_optimizer.zero_grad()
        #     for i, (task_name, task_config) in enumerate(self.task_cfg.tasks.items()):
        #         if self.no_train_last and i == len(self.task_cfg.tasks) - 1:
        #             break
        #         inputs = batch[i]
        #         if self.load_episode_vae:
        #             pdata, episode = inputs
        #             mu, logvar = self.episode_model.encode(episode)
        #             cond = self.episode_model.reparameterize(mu, logvar)
        #             loss += self.ae_forward(pdata, condition2=cond,**kwargs)
        #         else:
        #             if self.ae_use_condition:
        #                 loss += self.ae_forward(inputs, condition=torch.full((inputs.shape[0],), i).int().to(inputs.device), **kwargs)
        #             else:
        #                 loss += self.ae_forward(inputs, **kwargs)
        #     if self.no_train_last:
        #         loss /= len(self.task_cfg.tasks) - 1
        #     else:
        #         loss /= len(self.task_cfg.tasks)
        #     self.manual_backward(loss, retain_graph=True)
        #     ae_optimizer.step()
        #     self.log("ae_loss", loss.cpu().detach().mean().item(), on_epoch=True, prog_bar=True, logger=True)
        # else:
        loss = torch.tensor(0.0).cuda()
        ddpm_optimizer.zero_grad()
        for i, (task_name, task_config) in enumerate(self.task_cfg.tasks.items()):
            if self.no_train_last and i == len(self.task_cfg.tasks) - 1:
                break
            inputs = batch[i]
            if self.load_episode_vae:
                pdata, episode = inputs

                mu, logvar = self.episode_model.encode(episode)
                cond = self.episode_model.reparameterize(mu, logvar)
                loss += self.forward(pdata, cond=torch.full((pdata.shape[0],), i).int().to(pdata.device), cond2=cond, **kwargs)
            elif self.load_clip:
                pdata, episode = inputs
                #print("episode.shape: ", episode.shape)
                all_pil_images = []
                for episode_data in episode:
                    pil_images = [Image.fromarray(frame.cpu().numpy().astype(np.uint8)) for frame in episode_data]
                    all_pil_images.extend(pil_images)
                clip_inputs = self.processor(images=all_pil_images, return_tensors="pt")#.cuda()
                clip_inputs = {k: v.cuda() for k, v in clip_inputs.items()}
                #print(next(self.clip_model.parameters()).device)
                # for name, param in self.clip_model.named_parameters():
                #     print(f"{name} is on {param.device}")

                image_features = self.clip_model.get_image_features(**clip_inputs)
                cond = image_features.view(episode.shape[0], self.episode_len, -1).cuda()
                loss += self.forward(pdata, cond=torch.full((pdata.shape[0],), i).int().to(pdata.device), cond2=cond, **kwargs)
            elif self.load_vit:
                pdata, episode = inputs
                # 打印episode的形状以便调试
                # print(f"Episode shape: {episode.shape}")  # 应该是[batch_size, episode_len, h, w, c]
                
                batch_size = episode.shape[0]
                seq_len = episode.shape[1]
                
                # 为每个样本随机选择一帧
                random_indices = torch.randint(0, seq_len, (batch_size,))
                
                # 收集所有随机选择的帧
                random_frames = []
                for i in range(batch_size):
                    # 获取当前样本的随机帧
                    frame = episode[i, random_indices[i]]
                    # 转换为PIL图像
                    if frame.max() <= 1.0:  # 如果值在[0,1]范围内
                        frame = (frame * 255).byte()
                    frame_np = frame.cpu().numpy().astype(np.uint8)
                    
                    # 确保图像格式正确
                    if len(frame_np.shape) == 2:  # 灰度图
                        # 转换为RGB
                        frame_rgb = np.stack([frame_np] * 3, axis=2)
                        pil_img = Image.fromarray(frame_rgb, 'RGB')
                    elif len(frame_np.shape) == 3:
                        if frame_np.shape[2] == 1:  # 单通道
                            frame_rgb = np.repeat(frame_np, 3, axis=2)
                            pil_img = Image.fromarray(frame_rgb.squeeze(), 'RGB')
                        elif frame_np.shape[2] == 3:  # RGB
                            pil_img = Image.fromarray(frame_np, 'RGB')
                        elif frame_np.shape[2] == 4:  # RGBA
                            pil_img = Image.fromarray(frame_np, 'RGBA').convert('RGB')
                        else:
                            # 假设是HWC格式但通道数不是标准的
                            pil_img = Image.fromarray(frame_np[:,:,0], 'L').convert('RGB')
                    else:
                        # 不支持的格式，使用灰度图转换
                        print(f"警告: 遇到不支持的图像格式 {frame_np.shape}")
                        dummy = np.zeros((84, 84, 3), dtype=np.uint8)
                        pil_img = Image.fromarray(dummy, 'RGB')
                    
                    random_frames.append(pil_img)
                
                # 可以保存第一张图像用于调试
                # if random_frames:
                #     random_frames[0].save('/tmp/vit_input_sample.png')
                
                # 使用ViT处理器处理图像
                # try:
                # 先获取处理后的输入数据
                vit_inputs = self.vit_processor(images=random_frames, return_tensors="pt")
                
                # 分别处理不同类型的张量，保持正确的数据类型
                for key in vit_inputs:
                    if key == 'pixel_values':
                        # 像素值是浮点型
                        vit_inputs[key] = vit_inputs[key].to(pdata.device)
                    else:
                        # 其他输入可能是整型，使用long()确保类型正确
                        vit_inputs[key] = vit_inputs[key].long().to(pdata.device)
                
                # 获取ViT特征
                with torch.no_grad():
                    vit_output = self.vit_model(**vit_inputs)
                    vit_features = vit_output.last_hidden_state[:, 0]  # 使用CLS token作为特征
                    
                    # 使用特征降维层将768维特征降至64维
                    vit_features_reduced = self.vit_feature_reducer(vit_features)
                
                loss += self.forward(pdata, cond=None, cond2=vit_features_reduced, **kwargs)
                # except Exception as e:
                #     print(f"ViT处理出错: {e}")
                #     print(f"VIT输入的keys: {vit_inputs.keys() if 'vit_inputs' in locals() else '未创建'}")
                #     for key in vit_inputs.keys() if 'vit_inputs' in locals() else []:
                #         print(f"Key: {key}, 类型: {vit_inputs[key].dtype}, 形状: {vit_inputs[key].shape}")
                #     # 出错时退回到基本条件
                #     loss += self.forward(pdata, cond=torch.full((pdata.shape[0],), i).int().to(pdata.device), **kwargs)
            else:
                loss += self.forward(inputs, cond=torch.full((inputs.shape[0],), i).int().to(inputs.device), **kwargs)
        if self.no_train_last:
            loss /= len(self.task_cfg.tasks) - 1
        else:
            loss /= len(self.task_cfg.tasks)
        self.manual_backward(loss, retain_graph=True)
        ddpm_optimizer.step()
        self.log("ddpm_loss", loss.cpu().detach().mean().item(), on_epoch=True, prog_bar=True, logger=True)

        if hasattr(self, "lr_scheduler"):
            self.lr_scheduler.step()
        return {"loss": loss}

    def pre_process(self, batch, cond=None):
        latent = self.ae_model.encode(batch, condition=cond)
        self.latent_shape = latent.shape[-2:]
        return latent

    def post_process(self, outputs, cond=None):
        outputs = outputs.reshape(-1, *self.latent_shape)
        return self.ae_model.decode(outputs, condition=cond)

    def validation_step(self, batch, batch_idx, **kwargs: Any):
        if self.current_epoch == 0: return
        # if self.no_validate:
        #     if (self.current_epoch + 1) % self.save_interval == 0:
        #         filepath = os.path.join(self.dirpath, self.filename.format(epoch=self.current_epoch + 1))
        #         trainer.save_checkpoint(filepath)
        #         print(f"Checkpoint saved at epoch {epoch + 1} to {filepath}")
        #     return
        #if self.global_rank != 0: return
        #self.maybe_load_ae_model()
        if self.current_epoch <= self.split_epoch:
            if self.load_episode_vae:
                idx = np.random.choice(batch[0][0].shape[0], self.train_cfg.ddpm_eval_batch_size, replace=False)
                random_val_batch = [(b[0][idx], b[1][idx]) for b in batch]
            else:
                idx = np.random.choice(batch[0].shape[0], self.train_cfg.ae_eval_batch_size, replace=False)
                random_val_batch = [b[idx] for b in batch]
            dic = self.ae_validate_step(random_val_batch, self.train_cfg.ae_eval_batch_size)
        else:
            if self.load_episode_vae:
                idx = np.random.choice(batch[0][0].shape[0], self.train_cfg.ddpm_eval_batch_size, replace=False)
                random_val_batch = [(b[0][idx], b[1][idx]) for b in batch]
            elif self.load_clip:
                idx = np.random.choice(batch[0][0].shape[0], self.train_cfg.ddpm_eval_batch_size, replace=False)
                random_val_batch = [(b[0][idx], b[1][idx]) for b in batch]
            elif self.load_vit:
                idx = np.random.choice(batch[0][0].shape[0], self.train_cfg.ddpm_eval_batch_size, replace=False)
                random_val_batch = [(b[0][idx], b[1][idx]) for b in batch]
            else:
                idx = np.zeros(self.train_cfg.ddpm_eval_batch_size, dtype=int)
                random_val_batch = [b[idx] for b in batch]
            dic = self.ddpm_validate_step(random_val_batch, self.train_cfg.ddpm_eval_batch_size)
        return dic

    # def on_validation_epoch_end(self):
    #     # 在所有进程完成验证后进行同步
    #     self.trainer.strategy.barrier()
    
    def test_step(self, batch, batch_idx, **kwargs: Any):
        # 如果当前epoch小于等于split_epoch且trainer不在测试状态，则调用ae_validate_step函数
        if self.current_epoch <= self.split_epoch and not self.trainer.testing:
            #self.maybe_load_ae_model()
            dic = self.ae_validate_step(batch, 10)
        # 否则调用ddpm_validate_step函数
        else:
            #self.maybe_load_all_model()
            dic = self.ddpm_validate_step(batch, 20, save_param=False)
        return dic

    def ae_validate_step(self, batch, num):
        params = {}
        for i, (task_name, task_config) in enumerate(self.task_cfg.tasks.items()):
            if self.load_episode_vae:
                params[task_name], _ = batch[i]
            else:
                params[task_name] = batch[i]
        input_metrics = self.eval_params(params, num)

        print("Test AE params")
        ae_params = {}
        for i, (task_name, task_config) in enumerate(self.task_cfg.tasks.items()):
            #condition = torch.full((batch[i].shape[0],), i).int().to(batch[i].device)
            # latent = self.ae_model.encode(batch[i], condition=condition)
            # ae_params[task_name] = self.ae_model.decode(latent, condition=condition)
            if self.load_episode_vae:
                pdata, episode = batch[i]
                mu, logvar = self.episode_model.encode(episode)
                cond2 = self.episode_model.reparameterize(mu, logvar)
                latent = self.ae_model.encode(pdata, condition=cond2)
                ae_params[task_name] = self.ae_model.decode(latent, condition=cond2)
            else:
                condition = torch.full((batch[i].shape[0],), i).int().to(batch[i].device)
                if self.ae_use_condition:
                    latent = self.ae_model.encode(batch[i], condition=condition)
                else:
                    latent = self.ae_model.encode(batch[i])
                print("{} latent shape:{}".format(task_name, latent.shape))
                if self.ae_use_condition:
                    ae_params[task_name] = self.ae_model.decode(latent, condition=condition)
                else:
                    ae_params[task_name] = self.ae_model.decode(latent)
        ae_metrics = self.eval_params(ae_params, num)

        for task_name in ae_metrics.keys():
            mean_score = np.clip((ae_metrics[task_name].mean() - input_metrics[task_name].mean()) / np.abs(input_metrics[task_name].mean()) + 1, -1, 1.5)
            print(f"{task_name}: Input model best return: {np.max(input_metrics[task_name])}, AE model best return: {np.max(ae_metrics[task_name])}")
            print(f"{task_name}: Input model mean return: {np.mean(input_metrics[task_name])}, AE model mean return: {np.mean(ae_metrics[task_name])}, mean score: {mean_score}")
            print(f"{task_name}: Input model median return: {np.median(input_metrics[task_name])}, AE model median return: {np.median(ae_metrics[task_name])}")
            print(f"{task_name}: Input model std return: {np.std(input_metrics[task_name])}, AE model median return: {np.std(ae_metrics[task_name])}")
            print()
            self.log("task_score_" + task_name, torch.tensor(mean_score))

        all_task_score = np.mean(np.clip(
            [(ae_metrics[task_name].mean() - input_metrics[task_name].mean()) / np.abs(input_metrics[task_name].mean()) + 1 for task_name in ae_metrics.keys()], -1, 1.5))
        print("all_task_score: ", all_task_score)
        print("---------------------------------")
        self.log("mean_ae_acc", torch.tensor(all_task_score))
        self.log("mean_g_acc", -1.0)


    def ddpm_validate_step(self, batch, num, save_param=False):

        input_params = {}
        output_params = {}
        time1 = time.time()
        cfg_guidance_scale = getattr(self.config, "cfg_scale", 1.0) # Get from config or default to 1.0 (no guidance)
        print("cfg_guidance_scale: ", cfg_guidance_scale)
        
        for i, (task_name, task_config) in enumerate(self.task_cfg.tasks.items()):
            if self.validate_only_last_two and i < len(self.task_cfg.tasks) - 2:
                continue
            if self.load_episode_vae:
                input_params[task_name], episode = batch[i]
                #condition = torch.full((input_params[task_name].shape[0],), i).int().to(episode.device)
                #latent = self.pre_process(input_params[task_name], cond=condition)
                latent = input_params[task_name]
                if episode.shape[0] != num:
                    episode = episode[:num]
                mu, logvar = self.episode_model.encode(episode)
                cond = self.episode_model.reparameterize(mu, logvar)
                cur_batch = self.generate(latent, cond=cond, num=num, cfg_scale=cfg_guidance_scale)
                #outputs = self.post_process(cur_batch, cond=condition)
                output_params[task_name] = cur_batch
            elif self.load_clip:
                input_params[task_name], episode = batch[i]
                latent = input_params[task_name]
                all_pil_images = []
                for episode_data in episode:
                    pil_images = [Image.fromarray(frame.cpu().numpy().astype(np.uint8)) for frame in episode_data]
                    all_pil_images.extend(pil_images)
                clip_inputs = self.processor(images=all_pil_images, return_tensors="pt")  # .cuda()
                clip_inputs = {k: v.cuda() for k, v in clip_inputs.items()}
                image_features = self.clip_model.get_image_features(**clip_inputs)
                cond = image_features.view(episode.shape[0], self.episode_len, -1).cuda()

                cur_batch = self.generate(latent, cond=cond, num=num, cfg_scale=cfg_guidance_scale)
                output_params[task_name] = cur_batch
            elif self.load_vit:
                input_params[task_name], episode = batch[i]
                latent = input_params[task_name]
                if episode.shape[0] != num:
                    episode = episode[:num]
                
                # 保持与training_step一致的处理方式
                batch_size = episode.shape[0]
                seq_len = episode.shape[1]
                # print("batch_size: ", batch_size)
                
                # 为每个样本随机选择一帧
                random_indices = torch.randint(0, seq_len, (batch_size,))
                
                # 收集所有随机选择的帧
                random_frames = []
                for j in range(batch_size):
                    # 获取当前样本的随机帧
                    frame = episode[j, random_indices[j]]
                    # 转换为PIL图像
                    if frame.max() <= 1.0:  # 如果值在[0,1]范围内
                        frame = (frame * 255).byte()
                    frame_np = frame.cpu().numpy().astype(np.uint8)
                    
                    # 确保图像格式正确
                    if len(frame_np.shape) == 2:  # 灰度图
                        frame_rgb = np.stack([frame_np] * 3, axis=2)
                        pil_img = Image.fromarray(frame_rgb, 'RGB')
                    elif len(frame_np.shape) == 3:
                        if frame_np.shape[2] == 1:  # 单通道
                            frame_rgb = np.repeat(frame_np, 3, axis=2)
                            pil_img = Image.fromarray(frame_rgb.squeeze(), 'RGB')
                        elif frame_np.shape[2] == 3:  # RGB
                            pil_img = Image.fromarray(frame_np, 'RGB')
                        elif frame_np.shape[2] == 4:  # RGBA
                            pil_img = Image.fromarray(frame_np, 'RGBA').convert('RGB')
                        else:
                            pil_img = Image.fromarray(frame_np[:,:,0], 'L').convert('RGB')
                    else:
                        print(f"警告: 遇到不支持的图像格式 {frame_np.shape}")
                        dummy = np.zeros((84, 84, 3), dtype=np.uint8)
                        pil_img = Image.fromarray(dummy, 'RGB')
                    
                    # --- 新增：测试时随机擦除 ---
                    if hasattr(self, 'trainer') and self.trainer.testing and self.testing_erase and pil_img: # 检查 self.trainer 是否存在
                        try:
                            img_w, img_h = pil_img.size # 应该是 84, 84
                            area = img_w * img_h
                            target_area = area * self.erase_ratio

                            # 计算正方形边长
                            side_length = int(round(math.sqrt(target_area)))

                            # 确保边长不超过图像尺寸
                            side_length = min(side_length, img_w, img_h)

                            if side_length > 0: # 仅当边长大于0时才执行擦除
                                # 随机确定正方形的左上角坐标
                                top = random.randint(0, img_h - side_length)
                                left = random.randint(0, img_w - side_length)
                                print("top: ", top, "left: ", left, "side_length: ", side_length)

                                draw = ImageDraw.Draw(pil_img)
                                # 使用灰色进行填充
                                erase_color = (128, 128, 128)
                                draw.rectangle([left, top, left + side_length, top + side_length], fill=erase_color)
                        except Exception as e:
                             print(f"随机擦除失败: {e}")
                    # --- 结束：测试时随机擦除 ---


                    random_frames.append(pil_img)
                
                # 使用ViT处理器处理图像
                # try:
                # 先获取处理后的输入数据
                vit_inputs = self.vit_processor(images=random_frames, return_tensors="pt")
                
                # 分别处理不同类型的张量，保持正确的数据类型
                for key in vit_inputs:
                    if key == 'pixel_values':
                        # 像素值是浮点型
                        vit_inputs[key] = vit_inputs[key].to(latent.device)
                    else:
                        # 其他输入可能是整型，使用long()确保类型正确
                        vit_inputs[key] = vit_inputs[key].long().to(latent.device)
                
                # 获取ViT特征
                with torch.no_grad():
                    vit_output = self.vit_model(**vit_inputs)
                    vit_features = vit_output.last_hidden_state[:, 0]  # 使用CLS token作为特征
                
                # 使用特征降维层将768维特征降至64维
                vit_features_reduced = self.vit_feature_reducer(vit_features)
                
                # 使用降维后的特征
                cur_batch = self.generate(latent, cond=vit_features_reduced, num=num, cfg_scale=cfg_guidance_scale)
                output_params[task_name] = cur_batch
                # except Exception as e:
                #     print(f"验证时ViT处理出错: {e}")
                #     print(f"VIT输入的keys: {vit_inputs.keys() if 'vit_inputs' in locals() else '未创建'}")
                #     for key in vit_inputs.keys() if 'vit_inputs' in locals() else []:
                #         print(f"Key: {key}, 类型: {vit_inputs[key].dtype}, 形状: {vit_inputs[key].shape}")
                #     # 出错时退回到基本条件
                #     condition = torch.full((latent.shape[0],), i).int().to(latent.device)
                #     cur_batch = self.generate(latent, cond=condition, num=num)
                #     output_params[task_name] = cur_batch
            else:
                input_params[task_name] = batch[i]
                condition = torch.full((batch[i].shape[0],), i).int().to(batch[i].device)
                # latent = self.pre_process(input_params[task_name], cond=condition)
                latent = input_params[task_name]
                cur_batch = self.generate(latent, cond=condition, num=num, cfg_scale=cfg_guidance_scale)
                #outputs = self.post_process(cur_batch, cond=condition)
                output_params[task_name] = cur_batch
            if save_param == True:
                path = f"/home/llm_user/yxh/policy-diffusion/models/param_data/Atari_zoo/{task_name}/200"
                torch.save(output_params[task_name], f"{path}/generate_param.pt")


        time2 = time.time()
        print("generate time: ", time2 - time1)

        ddpm_metrics = self.eval_params(output_params, num)

        for task_name in ddpm_metrics.keys():
            mean_score = np.clip((ddpm_metrics[task_name].mean() - self.all_performance[task_name]["mean"]) / np.abs(self.all_performance[task_name]["mean"]) + 1, -1, 1.5)
            print(f"{task_name}: generated models best return: {np.max(ddpm_metrics[task_name])}")
            print(f"{task_name}: generated models mean return: {np.mean(ddpm_metrics[task_name])}, mean score: {mean_score}")
            print(f"{task_name}: generated models median return: {np.median(ddpm_metrics[task_name])}")
            print(f"{task_name}: generated models std return: {np.std(ddpm_metrics[task_name])}")
            print()
            self.log("task_score_" + task_name, torch.tensor(mean_score))

        all_task_score = np.mean(np.clip(
            [(np.mean(ddpm_metrics[task_name]) - self.all_performance[task_name]["mean"]) / np.abs(
                self.all_performance[task_name]["mean"]) + 1 for task_name in ddpm_metrics.keys()], -1, 1.5))
        print("all_task_score: ", all_task_score)
        print("---------------------------------")

        self.log('mean_g_acc', torch.tensor(all_task_score))
        self.log('mean_ae_acc', 0)
        return {'mean_g_acc': all_task_score}
    
    def eval_params(self, params, num):
        if self.validate_only_last_two:
            task_names = list(self.task_cfg.tasks.keys())[-2:]
        else:
            task_names = list(self.task_cfg.tasks.keys())
        metrics = {task_name: list() for task_name in task_names}

        print(f"Evaluating parameters ...")
        for i in tqdm(range(num)):
            result = self.task_func({task_name: params[task_name][i] for task_name in params.keys()})
            for task_name in metrics.keys():
                metrics[task_name].append(result[task_name])
        return {k: np.array(v) for k, v in metrics.items()}

    def eval_params_gg(self, params, num):
        metrics = {task_name: list() for task_name in self.task_cfg.tasks.keys()}
        print(f"Evaluating parameters ...")
        # 创建任务列表
        tasks = [{task_name: params[task_name][i] for task_name in params.keys()} for i in range(num)]

        # 使用 ProcessPoolExecutor 并行执行任务
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.task_func, task) for task in tasks]
            results = [future.result() for future in tqdm(futures)]

        # 收集结果
        for result in results:
            for task_name in metrics.keys():
                metrics[task_name].append(result[task_name])

        return {k: np.array(v) for k, v in metrics.items()}

    def forward(self, batch, cond=None, cond2=None, **kwargs):
        # if self.ae_use_condition:
        #     batch = self.pre_process(batch, cond)
        # else:
        #     batch = self.pre_process(batch)

        model = self.model
        time = (torch.rand(batch.shape[0]) * self.n_timestep).type(torch.int64).to(batch.device)

        noise = None
        lab = cond2
        if noise is None:
            noise = torch.randn_like(batch)
        x_t = self.q_sample(batch, time, noise=noise)

        # todo: loss using criterion, so we can change it
        if self.loss_type == 'kl':
            # the variational bound
            losses = self._vb_terms_bpd(model=model, x_0=batch, x_t=x_t, t=time, clip_denoised=False, return_pred_x0=False)

        elif self.loss_type == 'mse':
            # unweighted MSE
            assert self.model_var_type != 'learned'
            target = {
                'xprev': self.q_posterior_mean_variance(x_0=batch, x_t=x_t, t=time)[0],
                'xstart': batch,
                'eps': noise
            }[self.model_mean_type]
            model_output = model(x_t, time, cond=lab)
            losses = torch.mean((target - model_output).view(batch.shape[0], -1)**2, dim=1)

        else:
            raise NotImplementedError(self.loss_type)

        loss = losses.mean()

        # todo: ema is a insert
        if hasattr(self.model, 'ema'):
            accumulate(self.model.ema,
                       self.model.model if isinstance(self.model.model, nn.DataParallel) else self.model.model, 0.9999)

        self.log('train_loss', loss)
        return loss
        
    def configure_optimizers(self, **kwargs):
        #ae_parmas = self.ae_model.parameters()
        ddpm_params = self.model.parameters()

        self.ddpm_optimizer = hydra.utils.instantiate(self.train_cfg.optimizer, ddpm_params)
        #self.ae_optimizer = hydra.utils.instantiate(self.train_cfg.optimizer, ae_parmas)

        if "lr_scheduler" in self.train_cfg and self.train_cfg.lr_scheduler is not None:
            self.lr_scheduler = hydra.utils.instantiate(self.train_cfg.lr_scheduler)

        return self.ddpm_optimizer #self.ae_optimizer

    def maybe_load_ae_model(self):
        if self.current_epoch >= self.split_epoch - 1 and not self.load_best_ae and self.load_history_ae is None:
            # Load the best AE model
            ckpt_path = os.path.join(self.train_cfg.trainer.logger.save_dir, "checkpoints")
            possible_ae_paths = [os.path.join(ckpt_path, f) for f in os.listdir(ckpt_path) if "ae-epoch" in f]
            if len(possible_ae_paths) > 0:
                checkpoint = torch.load(possible_ae_paths[0])
                state_dict = {}
                for key in checkpoint["state_dict"]:
                    if "ae_model." in key:
                        new_key = key.replace("ae_model.", "") 
                        state_dict[new_key] = checkpoint["state_dict"][key]
                self.ae_model.load_state_dict(state_dict)
                print(f"Load the best AE model from {possible_ae_paths[0]}")
            else:
                print("Failed to load the best AE model from checkpoints, use the current AE model")
            self.load_best_ae = True

    def maybe_load_all_model(self):
        # Load the best all model
        ckpt_path = os.path.join(self.train_cfg.trainer.logger.save_dir, "checkpoints")
        possible_paths = [os.path.join(ckpt_path, f) for f in os.listdir(ckpt_path) if "ddpm-epoch" in f]
        if len(possible_paths) > 0:
            possible_paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            checkpoint = torch.load(possible_paths[0])
            state_dict_ae = {}
            state_dict_ddpm = {}
            for key in checkpoint["state_dict"]:
                #print(key)
                if key.startswith("ae_model."):
                    new_key = key.replace("ae_model.", "")
                    state_dict_ae[new_key] = checkpoint["state_dict"][key]
                elif key.startswith("model."):
                    new_key = key.replace("model.", "")
                    state_dict_ddpm[new_key] = checkpoint["state_dict"][key]
            self.ae_model.load_state_dict(state_dict_ae)
            self.model.load_state_dict(state_dict_ddpm)

            print(f"Load the best model from {possible_paths[0]}")
        else:
            print("Failed to load the best model from checkpoints, use the current model")
        self.load_best_ae = True
