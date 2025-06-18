import pdb

import hydra.utils
import pytorch_lightning as pl
import torch
from typing import Any
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from .base import BaseSystem
from core.utils.ddpm import *
from core.utils.utils import *
from core.module.prelayer.latent_transformer import Param2Latent
from .ddpm import DDPM
from core.module.modules.encoder import medium
import json

class MultitaskDDPM(DDPM):
    def __init__(self, config, **kwargs):
        self.task_cfg = config.task

        # for task_name, task_config in self.task_cfg.tasks.items():
        #     if hasattr(task_config, "train"):
        #
        #         config.system.ae_model.in_dim = getattr(task_config.train, "param_dim", config.system.ae_model.in_dim)
        #         #config.system.ae_model.in_dim = 1000
        #         print("config.system.ae_model.in_dim:", config.system.ae_model.in_dim)
        #         break

        ae_model = hydra.utils.instantiate(config.system.ae_model)

        # input_dim = config.system.ae_model.in_dim
        # input_noise = torch.randn((1, input_dim))
        # latent_dim = ae_model.encode(input_noise).shape
        # config.system.model.arch.model.in_dim = latent_dim[-1] * latent_dim[-2]

        super(MultitaskDDPM, self).__init__(config)
        self.save_hyperparameters()
        self.split_epoch = self.train_cfg.split_epoch
        self.ae_use_condition = self.train_cfg.ae_use_condition
        self.ddpm_use_condition = self.train_cfg.ddpm_use_condition
        self.loss_func = nn.MSELoss()
        self.ae_model = ae_model
        self.load_best_ae = False
        self.skip_tasks = getattr(self.task_cfg, "skip_tasks", None)

        self.load_history_ae = getattr(config.system, "ae_path", None)
        if getattr(config.system, "ae_path", None) is not None:
            checkpoint = torch.load(config.system.ae_path)
            new_state_dict = {}
            for key in checkpoint["state_dict"]:
                if "ae_model." in key:
                    new_key = key.replace("ae_model.", "")  # 假设所有的键都有一个不需要的'ae_model.'前缀
                    # if new_key.startswith('encoder.transformer_encoder_blocks.0.'):
                    #     new_key = new_key.replace('encoder.transformer_encoder_blocks.0.', 'encoder_blocks.0.transformer_encoder')
                    # elif new_key.startswith('decoder.transformer_encoder_blocks.0.'):
                    #     new_key = new_key.replace('decoder.transformer_decoder_blocks.0.', 'decoder_blocks.0.transformer_decoder')
                    new_state_dict[new_key] = checkpoint["state_dict"][key]


            # 打印模型的键
            # print("Model's state_dict:")
            # for key in ae_model.state_dict().keys():
            #     print(key)
            #
            # # 打印checkpoint的键
            # print("\nCheckpoint's state_dict:")
            # for key in checkpoint["state_dict"].keys():
            #     print(key)
            #
            # print("\nCheckpoint's new_state_dict:")
            # for key in new_state_dict.keys():
            #     print(key)
            ae_model.load_state_dict(new_state_dict)

        self.all_performance = {}
        for task_name, task_config in self.task_cfg.tasks.items():
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
            #self.all_performance[task_name] = performance

        print("all_preformance: ", self.all_performance)

        self.results_file = os.path.join(getattr(config, "output_dir", None), "results.json")
        print("results_path: ", self.results_file)
        if not os.path.exists(self.results_file):
            with open(self.results_file, 'w') as f:
                json.dump({}, f)

        ae_total_params = sum(p.numel() for p in self.ae_model.parameters())
        print(f'AE Total number of parameters: {ae_total_params}')
        ddpm_total_params = sum(p.numel() for p in self.model.parameters())
        print(f'ddpm Total number of parameters: {ddpm_total_params}')

    def ae_forward(self, batch, condition=None, **kwargs):
        output = self.ae_model(batch, condition=condition)
        loss = self.loss_func(batch, output, **kwargs)
        return loss

    def training_step(self, batch, batch_idx, **kwargs):
        ddpm_optimizer, ae_optimizer = self.optimizers()
        accumulate_loss = True
        if self.current_epoch <= self.split_epoch:
            if accumulate_loss:
                loss = torch.tensor(0.0).cuda()
                ae_optimizer.zero_grad()
                for i, (task_name, task_config) in enumerate(self.task_cfg.tasks.items()):
                    inputs = batch[i]
                    if self.ae_use_condition:
                        loss += self.ae_forward(inputs, condition=torch.full((inputs.shape[0],), i).int().to(inputs.device), **kwargs)
                    else:
                        loss += self.ae_forward(inputs, **kwargs)
                loss /= len(self.task_cfg.tasks)
                self.manual_backward(loss, retain_graph=True)
                ae_optimizer.step()
                self.log("ae_loss", loss.cpu().detach().mean().item(), on_epoch=True, prog_bar=True, logger=True)
            else:
                loss = 0.0
                for i, (task_name, task_config) in enumerate(self.task_cfg.tasks.items()):
                    inputs = batch[i]
                    print(i, task_name)
                    ae_optimizer.zero_grad()
                    if self.ae_use_condition:
                        cur_loss = self.ae_forward(inputs,
                                                condition=torch.full((inputs.shape[0],), i).int().to(inputs.device),
                                                **kwargs)
                    else:
                        cur_loss = self.ae_forward(inputs, **kwargs)
                    self.manual_backward(cur_loss, retain_graph=True)
                    ae_optimizer.step()
                #     loss += cur_loss
                # loss /= len(self.task_cfg.tasks)
                # self.log("ae_loss", loss.cpu().detach().mean().item(), on_epoch=True, prog_bar=True, logger=True)
        else:
            loss = torch.tensor(0.0).cuda()
            ddpm_optimizer.zero_grad()
            for i, (task_name, task_config) in enumerate(self.task_cfg.tasks.items()):
                if self.skip_tasks is not None and task_name in self.skip_tasks:
                    continue
                inputs = batch[i]
                #print(i, task_name, inputs.shape)
                loss += self.forward(inputs, cond=torch.full((inputs.shape[0],), i).int().to(inputs.device), **kwargs)
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
            idx = np.random.choice(batch[0].shape[0], min(self.train_cfg.ae_eval_batch_size, batch[0].shape[0]), replace=False)
            random_val_batch = [b[idx] for b in batch]
            dic = self.ae_validate_step(random_val_batch, min(self.train_cfg.ae_eval_batch_size, batch[0].shape[0]))
        else:
            #idx = np.random.choice(batch[0].shape[0], self.train_cfg.ddpm_eval_batch_size, replace=False)
            idx = np.zeros(self.train_cfg.ddpm_eval_batch_size, dtype=int)
            random_val_batch = [b[idx] for b in batch]
            dic = self.ddpm_validate_step(random_val_batch, self.train_cfg.ddpm_eval_batch_size)
        return dic
    
    def test_step(self, batch, batch_idx, **kwargs: Any):
        self.maybe_load_ae_model()
        if self.current_epoch <= self.split_epoch:
            # Test the full batch
            dic = self.ae_validate_step(batch, 200)
        else:
            #dic = super().test_step(batch, batch_idx, **kwargs)
            dic = self.ddpm_validate_step(batch, 200)
        return dic

    def ae_validate_step(self, batch, num):
        params = {}
        for i, (task_name, task_config) in enumerate(self.task_cfg.tasks.items()):
            params[task_name] = batch[i]
        input_metrics = self.eval_params(params, num)

        print("Test AE params")
        ae_params = {}
        for i, (task_name, task_config) in enumerate(self.task_cfg.tasks.items()):
            condition = torch.full((batch[i].shape[0],), i).int().to(batch[i].device)
            #print("{} batch shape:{}".format(task_name, batch[i].shape))
            if self.ae_use_condition:
                latent = self.ae_model.encode(batch[i], condition=condition)
            else:
                latent = self.ae_model.encode(batch[i])
            print("{} latent shape:{}".format(task_name, latent.shape))
            if self.ae_use_condition:
                ae_params[task_name] = self.ae_model.decode(latent, condition=condition)#.cpu()
            else:
                ae_params[task_name] = self.ae_model.decode(latent)#.cpu()
            print("{} ae params shape:{}".format(task_name, ae_params[task_name].shape))
        ae_metrics = self.eval_params(ae_params, num)

        if os.path.getsize(self.results_file) > 0:
            with open(self.results_file, 'r') as f:
                all_results = json.load(f)
        else:
            all_results = {}
        step_key = f"step_{self.global_step}"
        all_results[step_key] = {}
        for task_name in ae_metrics.keys():
            mean_score = np.clip((ae_metrics[task_name].mean() - input_metrics[task_name].mean()) / np.abs(input_metrics[task_name].mean()) + 1, -1, 1.5)
            print(f"{task_name}: Input model best return: {np.max(input_metrics[task_name])}, AE model best return: {np.max(ae_metrics[task_name])}")
            print(f"{task_name}: Input model mean return: {np.mean(input_metrics[task_name])}, AE model mean return: {np.mean(ae_metrics[task_name])}, mean score: {mean_score}")
            print(f"{task_name}: Input model median return: {np.median(input_metrics[task_name])}, AE model median return: {np.median(ae_metrics[task_name])}")
            print(f"{task_name}: Input model std return: {np.std(input_metrics[task_name])}, AE model median return: {np.std(ae_metrics[task_name])}")
            print()
            self.log("task_score_"+task_name, torch.tensor(mean_score))
            all_results[step_key][task_name] = {}
            all_results[step_key][task_name]["input"] = input_metrics[task_name].tolist()
            all_results[step_key][task_name]["ae"] = ae_metrics[task_name].tolist()
            all_results[step_key][task_name]["input_mean"] = float(np.mean(input_metrics[task_name]))
            all_results[step_key][task_name]["ae_mean"] = float(np.mean(ae_metrics[task_name]))
            all_results[step_key][task_name]["input_std"] = float(np.std(input_metrics[task_name]))
            all_results[step_key][task_name]["ae_std"] = float(np.std(ae_metrics[task_name]))
            all_results[step_key][task_name]["input_max"] = float(np.max(input_metrics[task_name]))
            all_results[step_key][task_name]["ae_max"] = float(np.max(ae_metrics[task_name]))
            all_results[step_key][task_name]["input_median"] = float(np.median(input_metrics[task_name]))
            all_results[step_key][task_name]["ae_median"] = float(np.median(ae_metrics[task_name]))
            all_results[step_key][task_name]["mean_score"] = float(mean_score)


        #all_task_score = np.mean([ae_metrics[task_name].mean() / input_metrics[task_name].mean() for task_name in ae_metrics.keys()])
        all_task_score = np.mean(np.clip(
            [(ae_metrics[task_name].mean() - input_metrics[task_name].mean()) / np.abs(input_metrics[task_name].mean()) + 1 for task_name in ae_metrics.keys()], -1, 1.5))
        print("all_task_score: ", all_task_score)
        print("---------------------------------")
        self.log("mean_ae_acc", torch.tensor(all_task_score))
        self.log("mean_g_acc", -5000.0)
        all_results[step_key]["all_task_score"] = all_task_score
        with open(self.results_file, 'w') as f:
            json.dump(all_results, f, indent=4)


    def ddpm_validate_step(self, batch, num):

        input_params = {}
        output_params = {}
        for i, (task_name, task_config) in enumerate(self.task_cfg.tasks.items()):
            if self.skip_tasks is not None and task_name in self.skip_tasks:
                continue
            input_params[task_name] = batch[i]
            condition = torch.full((batch[i].shape[0],), i).int().to(batch[i].device)
            #print(i, task_name, len(batch[i]))
            if self.ae_use_condition:
                latent = self.pre_process(input_params[task_name], cond=condition)
            else:
                latent = self.pre_process(input_params[task_name])
            # latent = self.ae_model.encode(input_params[task_name], condition=condition)
            # self.latent_shape = latent.shape[-2:]
            #print("latent.shape: ", latent.shape)
            if self.ddpm_use_condition:
                cur_batch = self.generate(latent, cond=condition, num=num)
            else:
                cur_batch = self.generate(latent, num=num)
            #print("cur_batch.shape: ", cur_batch.shape)
            if self.ae_use_condition:
                outputs = self.post_process(cur_batch, cond=condition)
            else:
                outputs = self.post_process(cur_batch)
            #print("outputs.shape: ", outputs.shape)
            # outputs = cur_batch.reshape(-1, *self.latent_shape)
            # outputs = self.ae_model.decode(outputs, condition=condition)

            #cur_batch = cur_batch.cpu()
            output_params[task_name] = outputs

        ddpm_metrics = self.eval_params(output_params, num)

        if os.path.getsize(self.results_file) > 0:
            with open(self.results_file, 'r') as f:
                all_results = json.load(f)
        else:
            all_results = {}
        step_key = f"step_{self.global_step}"
        all_results[step_key] = {}
        for task_name in ddpm_metrics.keys():
            mean_score = np.clip((ddpm_metrics[task_name].mean() - self.all_performance[task_name]["mean"]) / np.abs(self.all_performance[task_name]["mean"]) + 1, -1, 1.5)
            print(f"{task_name}: generated models best return: {np.max(ddpm_metrics[task_name])}")
            print(f"{task_name}: generated models mean return: {np.mean(ddpm_metrics[task_name])}, mean score: {mean_score}")
            print(f"{task_name}: generated models median return: {np.median(ddpm_metrics[task_name])}")
            print(f"{task_name}: generated models std return: {np.std(ddpm_metrics[task_name])}")
            print()
            self.log("task_score_" + task_name, torch.tensor(mean_score))

            all_results[step_key][task_name] = {}
            all_results[step_key][task_name]["ddpm"] = ddpm_metrics[task_name].tolist()
            all_results[step_key][task_name]["input_mean"] = self.all_performance[task_name]["mean"]
            all_results[step_key][task_name]["ddpm_mean"] = float(np.mean(ddpm_metrics[task_name]))
            all_results[step_key][task_name]["input_std"] = self.all_performance[task_name]["std"]
            all_results[step_key][task_name]["ddpm_std"] = float(np.std(ddpm_metrics[task_name]))
            all_results[step_key][task_name]["input_max"] = self.all_performance[task_name]["max"]
            all_results[step_key][task_name]["ddpm_max"] = float(np.max(ddpm_metrics[task_name]))
            all_results[step_key][task_name]["input_median"] = self.all_performance[task_name]["median"]
            all_results[step_key][task_name]["ddpm_median"] = float(np.median(ddpm_metrics[task_name]))
            all_results[step_key][task_name]["mean_score"] = float(mean_score)

        all_task_score = np.mean(np.clip(
            [(np.mean(ddpm_metrics[task_name]) - self.all_performance[task_name]["mean"]) / np.abs(
                self.all_performance[task_name]["mean"]) + 1 for task_name in ddpm_metrics.keys()], -1, 1.5))
        print("all_task_score: ", all_task_score)
        print("---------------------------------")

        self.log('mean_g_acc', torch.tensor(all_task_score))
        self.log('mean_ae_acc', 0)
        all_results[step_key]["all_task_score"] = float(all_task_score)
        with open(self.results_file, 'w') as f:
            json.dump(all_results, f, indent=4)
        return {'mean_g_acc': all_task_score}
    
    def eval_params(self, params, num):
        metrics = {task_name: list() for task_name in self.task_cfg.tasks.keys()}
        if self.skip_tasks is not None:
            for task_name in self.skip_tasks:
                metrics.pop(task_name)
        print(f"Evaluating parameters ...")
        for i in tqdm(range(num)):
            result = self.task_func({task_name: params[task_name][i] for task_name in params.keys()})
            for task_name in metrics.keys():
                metrics[task_name].append(result[task_name])
        return {k: np.array(v) for k, v in metrics.items()}

    def forward(self, batch, cond=None, **kwargs):
        if self.ae_use_condition:
            batch = self.pre_process(batch, cond)
        else:
            batch = self.pre_process(batch)
        #print("batch.shape: ", batch.shape)
        model = self.model
        time = (torch.rand(batch.shape[0]) * self.n_timestep).type(torch.int64).to(batch.device)

        noise = None
        lab = cond
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
            if self.ddpm_use_condition:
                model_output = model(x_t, time, cond=lab)
            else:
                model_output = model(x_t, time)
            losses       = torch.mean((target - model_output).view(batch.shape[0], -1)**2, dim=1)

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
