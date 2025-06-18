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

from .base import BaseSystem
from core.utils.ddpm import *
from core.utils.utils import *
from core.module.prelayer.latent_transformer import Param2Latent
from .ddpm import DDPM
from .multitask_ae_ddpm import MultitaskAE_DDPM
import json
from core.episode.episode_vae import EpisodeVAE

class MultitaskVQVAE_DDPM(MultitaskAE_DDPM):
    def __init__(self, config, **kwargs):
        self.task_cfg = config.task
        print(self.task_cfg)

        ae_model = hydra.utils.instantiate(config.system.ae_model)
        super(MultitaskVQVAE_DDPM, self).__init__(config)
        self.save_hyperparameters()
        self.split_epoch = self.train_cfg.split_epoch
        self.ae_use_condition = self.train_cfg.ae_use_condition
        self.loss_func = nn.MSELoss()
        self.ae_model = ae_model
        self.load_best_ae = False

        self.load_episode_vae = getattr(config.system, "load_episode_vae", None)
        self.no_train_last = getattr(config.system, "no_train_last", None)
        self.validate_only_last_two = getattr(config.system, "validate_only_last_two", None)
        if self.load_episode_vae:
            img_input_dim = 84 * 84 * 4  # Atari图像大小
            token_dim = 32
            hidden_dim = 256
            latent_dim = 64
            num_heads = 4
            num_layers = 2
            img_output_dim = 84 * 84 * 4  # Atari图像大小
            patch_size = 84
            episode_vae_path = os.path.join(self.config.project_root, "episode_outputs/5-1/vae-epoch=491-val_loss=0.16.ckpt")
            self.episode_model = EpisodeVAE(img_input_dim, token_dim, hidden_dim, latent_dim, num_heads, num_layers, img_output_dim, patch_size=patch_size, only_ae=True)
            checkpoint = torch.load(episode_vae_path)
            self.episode_model.load_state_dict(checkpoint['state_dict'])
            self.episode_model.eval()
            self.episode_model.cuda()

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


    def ae_forward(self, batch, condition=None, condition2=None, **kwargs):
        output, commit_loss = self.ae_model.forward(batch)
        reconstruction_loss = self.loss_func(batch, output, **kwargs)
        loss = reconstruction_loss + commit_loss
        # self.log("commit_loss", commit_loss)
        # self.log("reconstruction_loss", reconstruction_loss)
        return loss, reconstruction_loss, commit_loss

    def training_step(self, batch, batch_idx, **kwargs):
        ddpm_optimizer, ae_optimizer = self.optimizers()
        if self.current_epoch < self.split_epoch:
            loss = torch.tensor(0.0).cuda()
            commit_loss = torch.tensor(0.0).cuda()
            reconstruction_loss = torch.tensor(0.0).cuda()
            ae_optimizer.zero_grad()
            for i, (task_name, task_config) in enumerate(self.task_cfg.tasks.items()):
                if self.no_train_last and i == len(self.task_cfg.tasks) - 1:
                    break
                inputs = batch[i]
                if self.load_episode_vae:
                    pdata, episode = inputs
                    mu, logvar = self.episode_model.encode(episode)
                    cond = self.episode_model.reparameterize(mu, logvar)
                    loss += self.ae_forward(pdata, condition2=cond, **kwargs)
                else:
                    if self.ae_use_condition:
                        loss += self.ae_forward(inputs,
                                                condition=torch.full((inputs.shape[0],), i).int().to(inputs.device),
                                                **kwargs)
                    else:
                        cur_loss, cur_reconstruction_loss, cur_commit_loss = self.ae_forward(inputs, **kwargs)
                        loss += cur_loss
                        reconstruction_loss += cur_reconstruction_loss
                        commit_loss += cur_commit_loss
            if self.no_train_last:
                loss /= len(self.task_cfg.tasks) - 1
                reconstruction_loss /= len(self.task_cfg.tasks) - 1
                commit_loss /= len(self.task_cfg.tasks) - 1
            else:
                loss /= len(self.task_cfg.tasks)
                reconstruction_loss /= len(self.task_cfg.tasks)
                commit_loss /= len(self.task_cfg.tasks)
            self.manual_backward(loss, retain_graph=True)
            ae_optimizer.step()
            self.log("vqvae_loss", loss.cpu().detach().mean().item(), on_epoch=True, prog_bar=True, logger=True)
            self.log("commit_loss", commit_loss)
            self.log("reconstruction_loss", reconstruction_loss)
        else:
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
        self.maybe_load_ae_model()
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
            else:
                idx = np.zeros(self.train_cfg.ddpm_eval_batch_size, dtype=int)
                random_val_batch = [b[idx] for b in batch]
            dic = self.ddpm_validate_step(random_val_batch, self.train_cfg.ddpm_eval_batch_size)
        return dic
    
    def test_step(self, batch, batch_idx, **kwargs: Any):
        if self.current_epoch <= self.split_epoch and not self.trainer.testing:
            self.maybe_load_ae_model()
            dic = self.ae_validate_step(batch, 200)
        else:
            self.maybe_load_all_model()
            dic = self.ddpm_validate_step(batch, 200)
        return dic

    def ae_validate_step(self, batch, num):
        params = {}
        # for i, (task_name, task_config) in enumerate(self.task_cfg.tasks.items()):
        #     if self.validate_only_last_two and i < len(self.task_cfg.tasks) - 2:
        #         continue
        #     if self.load_episode_vae:
        #         params[task_name], _ = batch[i]
        #     else:
        #         params[task_name] = batch[i]
        # input_metrics = self.eval_params(params, num)

        print("Test AE params")
        ae_params = {}
        for i, (task_name, task_config) in enumerate(self.task_cfg.tasks.items()):
            if self.validate_only_last_two and i < len(self.task_cfg.tasks) - 2:
                continue
            if self.load_episode_vae:
                pdata, episode = batch[i]
                mu, logvar = self.episode_model.encode(episode)
                cond2 = self.episode_model.reparameterize(mu, logvar)
                quantized, vq_loss = self.ae_model.encode(pdata, condition=cond2)
                ae_params[task_name] = self.ae_model.decode(quantized, condition=cond2)
            else:
                condition = torch.full((batch[i].shape[0],), i).int().to(batch[i].device)
                if self.ae_use_condition:
                    quantized, vq_loss = self.ae_model.encode(batch[i], condition=condition)
                else:
                    quantized, vq_loss = self.ae_model.encode(batch[i])
                #print("{} quantized shape:{}".format(task_name, quantized.shape))
                if self.ae_use_condition:
                    ae_params[task_name] = self.ae_model.decode(quantized, condition=condition)
                else:
                    ae_params[task_name] = self.ae_model.decode(quantized)
                reconstruction_loss = self.loss_func(batch[i], ae_params[task_name])
                self.log("val_commit_loss_" + task_name, torch.tensor(vq_loss))
                self.log("val_reconstruction_loss_" + task_name, torch.tensor(reconstruction_loss))
        ae_metrics = self.eval_params(ae_params, num)

        # for task_name in ae_metrics.keys():
        #     mean_score = np.clip((ae_metrics[task_name].mean() - input_metrics[task_name].mean()) / np.abs(
        #         input_metrics[task_name].mean()) + 1, -1, 1.5)
        #     print(
        #         f"{task_name}: Input model best return: {np.max(input_metrics[task_name])}, AE model best return: {np.max(ae_metrics[task_name])}")
        #     print(
        #         f"{task_name}: Input model mean return: {np.mean(input_metrics[task_name])}, AE model mean return: {np.mean(ae_metrics[task_name])}, mean score: {mean_score}")
        #     print(
        #         f"{task_name}: Input model median return: {np.median(input_metrics[task_name])}, AE model median return: {np.median(ae_metrics[task_name])}")
        #     print(
        #         f"{task_name}: Input model std return: {np.std(input_metrics[task_name])}, AE model std return: {np.std(ae_metrics[task_name])}")
        #     print()
        #     self.log("task_score_" + task_name, torch.tensor(mean_score))
        #
        # all_task_score = np.mean(np.clip(
        #     [(ae_metrics[task_name].mean() - input_metrics[task_name].mean()) / np.abs(
        #         input_metrics[task_name].mean()) + 1 for task_name in ae_metrics.keys()], -1, 1.5))

        for task_name in ae_metrics.keys():
            mean_score = np.clip((ae_metrics[task_name].mean() - self.all_performance[task_name]["mean"]) / np.abs(self.all_performance[task_name]["mean"]) + 1, -1, 1.5)
            print(f"{task_name}: AE model best return: {np.max(ae_metrics[task_name])}")
            print(f"{task_name}: AE model mean return: {np.mean(ae_metrics[task_name])}, mean score: {mean_score}")
            print(f"{task_name}: AE model median return: {np.median(ae_metrics[task_name])}")
            print(f"{task_name}: AE model std return: {np.std(ae_metrics[task_name])}")
            print()
            self.log("task_score_" + task_name, torch.tensor(mean_score))

        all_task_score = np.mean(np.clip(
            [(np.mean(ae_metrics[task_name]) - self.all_performance[task_name]["mean"]) / np.abs(
                self.all_performance[task_name]["mean"]) + 1 for task_name in ae_metrics.keys()], -1, 1.5))


        print("all_task_score: ", all_task_score)
        print("---------------------------------")
        self.log("mean_ae_acc", torch.tensor(all_task_score))
        self.log("mean_g_acc", -1.0)


    def ddpm_validate_step(self, batch, num):

        input_params = {}
        output_params = {}
        time1 = time.time()
        for i, (task_name, task_config) in enumerate(self.task_cfg.tasks.items()):
            if self.load_episode_vae:
                input_params[task_name], episode = batch[i]
                condition = torch.full((input_params[task_name].shape[0],), i).int().to(episode.device)
                latent = self.pre_process(input_params[task_name], cond=condition)

                mu, logvar = self.episode_model.encode(episode)
                cond = self.episode_model.reparameterize(mu, logvar)
                cur_batch = self.generate(latent, cond=cond, num=num)
                outputs = self.post_process(cur_batch, cond=condition)
                output_params[task_name] = outputs
            else:
                input_params[task_name] = batch[i]
                condition = torch.full((batch[i].shape[0],), i).int().to(batch[i].device)
                latent = self.pre_process(input_params[task_name], cond=condition)
                cur_batch = self.generate(latent, cond=condition, num=num)
                outputs = self.post_process(cur_batch, cond=condition)
                output_params[task_name] = outputs

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
        if self.ae_use_condition:
            batch = self.pre_process(batch, cond)
        else:
            batch = self.pre_process(batch)

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
        ae_parmas = self.ae_model.parameters()
        ddpm_params = self.model.parameters()

        self.ddpm_optimizer = hydra.utils.instantiate(self.train_cfg.optimizer, ddpm_params)
        self.ae_optimizer = hydra.utils.instantiate(self.train_cfg.optimizer, ae_parmas)

        if "lr_scheduler" in self.train_cfg and self.train_cfg.lr_scheduler is not None:
            self.lr_scheduler = hydra.utils.instantiate(self.train_cfg.lr_scheduler)

        return self.ddpm_optimizer, self.ae_optimizer

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
