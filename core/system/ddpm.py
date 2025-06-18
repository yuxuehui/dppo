import pdb

import hydra.utils
import pytorch_lightning as pl
import torch
from typing import Any
import numpy as np
import torch.nn as nn

from .base import BaseSystem
from core.utils.ddpm import *
from core.utils.utils import *
from core.module.prelayer.latent_transformer import Param2Latent


class DDPM(BaseSystem):
    def __init__(self, config, **kwargs):
        super(DDPM, self).__init__(config)
        self.save_hyperparameters()
        betas = self.config.beta_schedule
        self.n_timestep = betas.n_timestep
        betas = make_beta_schedule(**betas)
        self.betas_register(betas)

    def betas_register(self, betas):
        model_mean_type = self.config.model_mean_type
        model_var_type = self.config.model_var_type
        betas = betas.type(torch.float64)
        timesteps = betas.shape[0]
        self.num_timesteps = int(timesteps)

        self.model_mean_type = model_mean_type  # xprev, xstart, eps
        self.model_var_type = model_var_type  # learned, fixedsmall, fixedlarge
        self.loss_type = self.config.loss_type  # kl, mse

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        alphas_cumprod_prev = torch.cat(
            (torch.tensor([1], dtype=torch.float64), alphas_cumprod[:-1]), 0
        )
        # Calculate alphas_cumprod_next using torch tensors instead of numpy
        alphas_cumprod_next = torch.cat([alphas_cumprod[1:], torch.tensor([0.0], dtype=torch.float64)], dim=0)
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

        self.register("betas", betas)
        self.register("alphas_cumprod", alphas_cumprod)
        self.register("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register("alphas_cumprod_next", alphas_cumprod_next)
        
        self.register("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))
        self.register("log_one_minus_alphas_cumprod", torch.log(1 - alphas_cumprod))
        self.register("sqrt_recip_alphas_cumprod", torch.rsqrt(alphas_cumprod))
        self.register("sqrt_recipm1_alphas_cumprod", torch.sqrt(1 / alphas_cumprod - 1))
        self.register("posterior_variance", posterior_variance)
        self.register("posterior_log_variance_clipped",
                      torch.log(torch.cat((posterior_variance[1].view(1, 1),
                                           posterior_variance[1:].view(-1, 1)), 0)).view(-1))
        self.register("posterior_mean_coef1", (betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod)))
        self.register("posterior_mean_coef2", ((1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod)))

    def generate(self, batch, num=10, history=False, cond=None, cfg_scale=1.0):
        """Generate samples using either DDIM or regular sampling based on config
        """
        model = self.model.ema if hasattr(self.model, 'ema') else self.model
        model.eval()
        if len(batch.shape) > 2:
            shape = (num, 1, batch.shape[1] * batch.shape[2])
        else:
            shape = (num, 1, batch.shape[1])
        shape = (num, *batch.shape[1:])
        sample = self.progressive_samples_fn_simple(
            model,
            shape,
            device='cuda',
            cond=cond,
            cfg_scale=cfg_scale,
            include_x0_pred_freq=50,
            history=history,
        )

        if history:
            return sample['samples'], sample['history']
        return sample['samples']


    def progressive_samples_fn_simple(self, model, shape, device, cond=None, cfg_scale=1.0, include_x0_pred_freq=50, history=False):
        if getattr(self.config, 'sampling_method', 'ddim') == 'ddim':
            ## Note!! you can define a "cond_fn" to calculate the conditional score, now we are using the implicit classifier.
            ## you need to feed the cond_fn to the model, and the model will use it to calculate the conditional score.
            print("using ddim sampling")
            samples, history_data = self.ddim_sample_loop_progressive(
                model=model,
                shape=shape,
                device=device,
                cond=cond,
                cond_fn=None,
                cfg_scale=cfg_scale,
                noise_fn=torch.randn,
                include_x0_pred_freq=include_x0_pred_freq,
            )
        else:  # p_sample
            print("using p_sample sampling")
            samples, history_data = self.p_sample_loop_progressive_simple(
                model=model,
                shape=shape,
                cond=cond,
                cfg_scale=cfg_scale,
                noise_fn=torch.randn,
                device=device,
                include_x0_pred_freq=include_x0_pred_freq,
            )
        if history:
            return {'samples': samples, 'history': history_data}
        return {'samples': samples}

    def pre_process(self, batch, cond=None):
        if hasattr(self, 'data_transform') and self.data_transform is not None:
            batch = self.data_transform.pre_process(batch)
        return batch

    def post_process(self, outputs, cond=None):
        if hasattr(self, 'data_transform') and self.data_transform is not None:
            outputs = self.data_transform.post_process(outputs)
        return outputs

    def validation_step(self, batch, batch_idx, **kwargs: Any):
        batch = self.pre_process(batch)
        outputs = self.generate(batch, num=10)

        params = self.post_process(outputs)
        params = params.cpu()

        accs = []
        for i in range(params.shape[0]):
            param = params[i].to(batch.device)
            acc, test_loss, output_list = self.task_func(param)
            accs.append(acc)
        best_acc = np.max(accs)
        print("generated models accuracy:", accs)
        print("generated models mean accuracy:", np.mean(accs))
        print("generated models best accuracy:", best_acc)
        print("generated models median accuracy:", np.median(accs))
        self.log('best_g_acc', best_acc)
        self.log('mean_g_acc', np.mean(accs).item())
        self.log('med_g_acc', np.median(accs).item())
        return {'best_g_acc': best_acc, 'mean_g_acc': np.mean(accs).item()}

    def test_step(self, batch, batch_idx, **kwargs: Any):
        # Load best ddpm model
        ckpt_path = os.path.join(self.train_cfg.trainer.logger.save_dir, "checkpoints")
        possible_ddpm_paths = [os.path.join(ckpt_path, f) for f in os.listdir(ckpt_path) if "ddpm-epoch" in f]
        if len(possible_ddpm_paths) > 0:
            checkpoint = torch.load(possible_ddpm_paths[0])
            self.load_state_dict(checkpoint["state_dict"])
            print(f"Load the best DDPM model from {possible_ddpm_paths[0]}")
        else:
            print("Failed to load the best ddpm model, use current model for testing")

        # generate 200 models
        batch = self.pre_process(batch)
        outputs = self.generate(batch, num=20)
        params = self.post_process(outputs)
        accs = []
        for i in range(params.shape[0]):
            param = params[i]
            acc, test_loss, output_list = self.task_func(param)
            accs.append(acc)
        best_acc = np.max(accs)
        print("generated models accuracy:", accs)
        print("generated models mean accuracy:", np.mean(accs))
        print("generated models best accuracy:", best_acc)
        print("generated models median accuracy:", np.median(accs))
        self.log('best_g_acc', best_acc)
        self.log('mean_g_acc', np.mean(accs).item())
        self.log('med_g_acc', np.median(accs).item())
        return {'best_g_acc': best_acc, 'mean_g_acc': np.mean(accs).item(), 'med_g_acc': np.median(accs).item()}

    def forward(self, batch, cond=None, **kwargs):
        """
        Compute training losses for random timesteps.
        :param batch: the clean data
        :param cond: the conditional input
        """
        batch = self.pre_process(batch, cond)
        model = self.model
        time = (torch.rand(batch.shape[0]) * self.n_timestep).type(torch.int64).to(batch.device)

        noise = None
        lab = cond
        if noise is None:
            noise = torch.randn_like(batch)
        x_t = self.q_sample(batch, time, noise=noise)   # the forward process x_t = √(αₜ) * x_0 + √(1 - αₜ) * ε

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


    def register(self, name, tensor):
        self.register_buffer(name, tensor.type(torch.float32))

    def _prior_bpd(self, x_0):

        B, T                        = x_0.shape[0], self.num_timesteps
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_0,
                                                           t=torch.full((B,), T - 1, dtype=torch.int64))
        kl_prior                    = normal_kl(mean1=qt_mean,
                                                logvar1=qt_log_variance,
                                                mean2=torch.zeros_like(qt_mean),
                                                logvar2=torch.zeros_like(qt_log_variance))

        return torch.mean(kl_prior.view(B, -1), dim=1)/np.log(2.)

    @torch.no_grad()
    def calc_bpd_loop(self, model, x_0, clip_denoised):

        (B, C, H, W), T = x_0.shape, self.num_timesteps

        new_vals_bt = torch.zeros((B, T))
        new_mse_bt  = torch.zeros((B, T))

        for t in reversed(range(self.num_timesteps)):

            t_b = torch.full((B, ), t, dtype=torch.int64)

            # Calculate VLB term at the current timestep
            new_vals_b, pred_x0 = self._vb_terms_bpd(model=model,
                                                     x_0=x_0,
                                                     x_t=self.q_sample(x_0=x_0, t=t_b),
                                                     t=t_b,
                                                     clip_denoised=clip_denoised,
                                                     return_pred_x0=True)

            # MSE for progressive prediction loss
            new_mse_b = torch.mean((pred_x0-x_0).view(B, -1)**2, dim=1)

            # Insert the calculated term into the tensor of all terms
            mask_bt = (t_b[:, None] == torch.arange(T)[None, :]).to(torch.float32)

            new_vals_bt = new_vals_bt * (1. - mask_bt) + new_vals_b[:, None] * mask_bt
            new_mse_bt  = new_mse_bt  * (1. - mask_bt) + new_mse_b[:, None] * mask_bt

        prior_bpd_b = self._prior_bpd(x_0)
        total_bpd_b = torch.sum(new_vals_bt, dim=1) + prior_bpd_b

        return total_bpd_b, new_vals_bt, prior_bpd_b, new_mse_bt

    def q_mean_variance(self, x_0, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
        variance = extract(1. - self.alphas_cumprod, t, x_0.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_0.shape)
        return mean, variance, log_variance

    def q_sample(self, x_0, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        x_t = √(αₜ) * x_0 + √(1 - αₜ) * ε
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        return (extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
                + extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise)


    def q_posterior_mean_variance(self, x_0, x_t, t):
        mean            = (extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
                           + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        var             = extract(self.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return mean, var, log_var_clipped

    # def p_mean_variance(self, model, x, t, clip_denoised, return_pred_x0, lab):
    def p_mean_variance(self, model, x, t, clip_denoised, return_pred_x0, cond=None):

        # import pdb; pdb.set_trace()
        # model_output = model(x, t, cond=lab)
        model_output = model(x, t, cond=cond)


        # Learned or fixed variance?
        if self.model_var_type == 'learned':
            model_output, log_var = torch.split(model_output, 2, dim=-1)
            var                   = torch.exp(log_var)

        elif self.model_var_type in ['fixedsmall', 'fixedlarge']:

            # below: only log_variance is used in the KL computations
            var, log_var = {
                # for 'fixedlarge', we set the initial (log-)variance like so to get a better decoder log likelihood
                'fixedlarge': (self.betas, torch.log(torch.cat((self.posterior_variance[1].view(1, 1),
                                                                self.betas[1:].view(-1, 1)), 0)).view(-1)),
                'fixedsmall': (self.posterior_variance, self.posterior_log_variance_clipped),
            }[self.model_var_type]
            # import pdb;pdb.set_trace()

            var     = extract(var, t, x.shape) * torch.ones_like(x)
            log_var = extract(log_var, t, x.shape) * torch.ones_like(x)
        else:
            raise NotImplementedError(self.model_var_type)

        # Mean parameterization
        _maybe_clip = lambda x_: (x_.clamp(min=-1, max=1) if clip_denoised else x_)

        if self.model_mean_type == 'xprev':
            # the model predicts x_{t-1}
            pred_x_0 = _maybe_clip(self.predict_start_from_prev(x_t=x, t=t, x_prev=model_output))
            mean     = model_output
        elif self.model_mean_type == 'xstart':
            # the model predicts x_0
            pred_x0    = _maybe_clip(model_output)
            mean, _, _ = self.q_posterior_mean_variance(x_0=pred_x0, x_t=x, t=t)
        elif self.model_mean_type == 'eps':
            # the model predicts epsilon
            pred_x0    = _maybe_clip(self.predict_start_from_noise(x_t=x, t=t, noise=model_output))
            mean, _, _ = self.q_posterior_mean_variance(x_0=pred_x0, x_t=x, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        if return_pred_x0:
            return mean, var, log_var, pred_x0
        else:
            return mean, var, log_var

    def predict_start_from_noise(self, x_t, t, noise):
        # import pdb; pdb.set_trace()

        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

    def predict_start_from_prev(self, x_t, t, x_prev):

        return (extract(1./self.posterior_mean_coef1, t, x_t.shape) * x_prev -
                extract(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape) * x_t)

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    

    def condition_score(self, model, pred_x0, x, t, cond, cfg_scale=1.0, cond_fn=None, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.
        See condition_mean() for details on cond_fn.
        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        if cond_fn is not None:
            alpha_bar = extract(self.alphas_cumprod, t, x.shape)

            eps = self._predict_eps_from_xstart(x, t, pred_x0)
            eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, **model_kwargs)
            # scaling_factor = sqrt(1 - α̅t)
            # conditional_gradient = ∇x log(p(y|x))
        else:
            alpha_bar = extract(self.alphas_cumprod, t, x.shape)

            eps = self._predict_eps_from_xstart(x, t, pred_x0)
            # if use implicit classifier
            # 1. Prepare unconditional input (zero vector)
            cond_uncond = torch.zeros_like(cond)

            # 2. Combine inputs for batched computation
            x_combined = torch.cat([x, x], dim=0)           # Shape (2B, ...)
            t_combined = torch.cat([t, t], dim=0)           # Shape (2B,)
            cond_combined = torch.cat([cond, cond_uncond], dim=0) # Shape (2B, D)

            # 3. Call model once
            #    IMPORTANT ASSUMPTION: model directly predicts noise 'eps'
            if self.model_mean_type != 'eps':
                # If model predicts x_0 or x_{t-1}, need to derive eps first
                # For now, raise error if assumption is violated
                raise NotImplementedError(f"CFG implementation currently assumes model_mean_type='eps', but got '{self.model_mean_type}'")
            model_output_combined = model(x_combined, t_combined, cond=cond_combined) # Shape (2B, ...) should be eps prediction

            # 4. Separate conditional and unconditional outputs
            cond_eps, uncond_eps = model_output_combined.chunk(2, dim=0) # Each shape (B, ...)

            # 5. Calculate guided noise prediction
            # Formula: eps = uncond_eps + scale * (cond_eps - uncond_eps)
            guided_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps) ### grad(log(p(y|x)))

            eps = eps - (1 - alpha_bar).sqrt() * guided_eps

        new_pred_x0 = self._predict_xstart_from_eps(x, t, eps)
        new_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_x0, x_t=x, t=t)
        return new_pred_x0, new_mean

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        cond_fn=None,
        cond=None,
        cfg_scale=1.0,
        eta=0.0,
        return_pred_x0=False
    ):
        """
        Sample x_{t-1} from the model using DDIM sampling.
        Args:
            model: The model to sample from
            x: Current tensor at time t
            t: Current timestep
            clip_denoised: If True, clip denoised samples to [-1, 1]
            cond: Conditional input for guided sampling
            cfg_scale: Classifier-free guidance scale (1.0 means no guidance)
            eta: DDIM eta parameter (0.0 means deterministic sampling)
        Returns:
            dict: Contains 'sample' (x_{t-1}) and 'pred_x0' predictions
        """
        # Get model predictions
        model_mean, model_variance, model_log_variance, pred_x0 = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            cond=cond,
            return_pred_x0=True
        )
        if cond_fn is not None and cfg_scale > 1.0:
            # Apply guidance
            model_mean, pred_x0 = self.condition_score(model, model_mean, pred_x0, x, t, cond, cfg_scale, cond_fn=cond_fn,)

        # Extract required values
        alpha_bar = extract(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t, x.shape)
        
        # Calculate sigma (η in DDIM paper)
        sigma = eta * torch.sqrt(
            (1 - alpha_bar_prev) / (1 - alpha_bar)
        ) * torch.sqrt(1 - alpha_bar / alpha_bar_prev)

        # Get epsilon prediction
        eps = self._predict_eps_from_xstart(x, t, pred_x0)

        # DDIM mean prediction (deterministic part)
        mean_pred = (
            pred_x0 * torch.sqrt(alpha_bar_prev) +
            torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps
        )

        # Add noise if eta > 0
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise

        return (sample, pred_x0) if return_pred_x0 else sample

    def ddim_sample_loop(
        self,
        model,
        shape,
        device,
        noise_fn=torch.randn,
        cond_fn=None,
        cond=None,
        cfg_scale=1.0
    ):
        """
        Generate samples from the model using DDIM.
        Same usage as p_sample_loop().
        """
        device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
        # shape[0] = lab.shape[0]
        img = noise_fn(shape).to(device)

        indices = list(range(self.num_timesteps))[::-1]

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                img = self.ddim_sample(
                    model,
                    img,
                    t,
                    cond_fn=cond_fn,
                    cond=cond,
                    return_pred_x0=False,
                    cfg_scale=cfg_scale,
                )

        return img

    def ddim_sample_loop_progressive(
        self, 
        model, 
        shape, 
        device, 
        cond=None, 
        cond_fn=None,
        cfg_scale=1.0, 
        noise_fn=torch.randn, 
        include_x0_pred_freq=50
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.
        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))

        img = noise_fn(shape, dtype=torch.float32).to(device)
        num_recorded_x0_pred = self.num_timesteps // include_x0_pred_freq
        x0_preds_ = torch.zeros((shape[0], num_recorded_x0_pred, *shape[1:]), dtype=torch.float32).to(device)


        indices = list(range(self.num_timesteps))[::-1]

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                sample_output, pred_x0 = self.ddim_sample(
                    model,
                    img,
                    t,
                    cond_fn=cond_fn,
                    cond=cond,
                    return_pred_x0=True
                )
            img = sample_output # Update img with the sampled output

            # Keep track of prediction of x0
            if i % include_x0_pred_freq == 0 or i == self.num_timesteps - 1: # Record more frequently or adjust logic
                idx = i // include_x0_pred_freq
                if 0 <= idx < num_recorded_x0_pred:
                     x0_preds_[:, idx, ...] = pred_x0
        
        return img, x0_preds_

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        cond_fn=None,
        cond=None,
        cfg_scale=1.0,
        eta=0.0,
        return_pred_x0=False
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        
        model_mean, model_variance, model_log_variance, pred_x0 = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            cond=cond,
            return_pred_x0=True
        )
        if cond_fn is not None and cfg_scale > 1.0:
            # Apply guidance
            model_mean, pred_x0 = self.condition_score(model, model_mean, pred_x0, x, t, cond, cfg_scale, cond_fn=cond_fn,)
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - pred_x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = extract(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = pred_x0 * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * eps

        return mean_pred, pred_x0


    # def p_sample(self, model, x, t, noise_fn, clip_denoised=True, return_pred_x0=False, lab=None):
    def p_sample(self, model, x, t, noise_fn, cond=None, cfg_scale=1.0, clip_denoised=True, return_pred_x0=False):
        """
        Sample the model for one step.

        :param model: the model to use.
        :param x: the current tensor at timestep t.
        :param t: the value of t, starting at 0 for the first-noise step.
        :param noise_fn: a function that generates noise.
        :param cond: the conditional input. If provided and cfg_scale > 1.0, CFG is applied.
        :param cfg_scale: the scale for classifier-free guidance.
        :param clip_denoised: if True, clip the denoised signal to [-1, 1].
        :param return_pred_x0: if True, return the predicted x_0 as well.
        :return: a dict containing the following keys:
                 - 'sample': the sampled tensor (shape like x).
                 - 'pred_x0': the predicted x_0 (if return_pred_x0 is True).
        """
        # pdb.set_trace()
        B = x.shape[0] # Original batch size

        # --- Classifier-Free Guidance Logic ---
        if cond is not None and cfg_scale > 1.0 and False:
            ### batches the unconditional forward pass for classifier-free guidance.
            # cfg_scale: the scale factor for classifier-free guidance. gamma in the paper.
            # 1. Prepare unconditional input (zero vector)
            cond_uncond = torch.zeros_like(cond)

            # 2. Combine inputs for batched computation
            x_combined = torch.cat([x, x], dim=0)           # Shape (2B, ...)
            t_combined = torch.cat([t, t], dim=0)           # Shape (2B,)
            cond_combined = torch.cat([cond, cond_uncond], dim=0) # Shape (2B, D)

            # 3. Call model once
            #    IMPORTANT ASSUMPTION: model directly predicts noise 'eps'
            if self.model_mean_type != 'eps':
                # If model predicts x_0 or x_{t-1}, need to derive eps first
                # For now, raise error if assumption is violated
                raise NotImplementedError(f"CFG implementation currently assumes model_mean_type='eps', but got '{self.model_mean_type}'")
            model_output_combined = model(x_combined, t_combined, cond=cond_combined) # Shape (2B, ...) should be eps prediction

            # 4. Separate conditional and unconditional outputs
            cond_eps, uncond_eps = model_output_combined.chunk(2, dim=0) # Each shape (B, ...)

            # 5. Calculate guided noise prediction
            # Formula: eps = uncond_eps + scale * (cond_eps - uncond_eps)
            guided_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps) ### grad(log(p(y|x)))

            # 6. Calculate pred_x0 using the guided noise
            # Use original x and t, but guided_eps
            pred_x0 = self.predict_start_from_noise(x, t, guided_eps)

            if clip_denoised:
                pred_x0 = pred_x0.clamp(min=-1, max=1)

            # 7. Calculate model mean using the guided pred_x0
            model_mean, _, model_log_variance = self.q_posterior_mean_variance(x_0=pred_x0, x_t=x, t=t)

        elif cond is not None and cfg_scale > 1.0:
            model_mean, model_variance, model_log_variance, pred_x0 = self.p_mean_variance(
                model, x, t, clip_denoised=clip_denoised, cond=cond, return_pred_x0=True
            )
            ### Then we need to shift the model_mean through classifier-free guidance

            ### This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
            ### batches the unconditional forward pass for classifier-free guidance.
            # cfg_scale: the scale factor for classifier-free guidance. gamma in the paper.
            # 1. Prepare unconditional input (zero vector)
            cond_uncond = torch.zeros_like(cond)

            # 2. Combine inputs for batched computation
            x_combined = torch.cat([x, x], dim=0)           # Shape (2B, ...)
            t_combined = torch.cat([t, t], dim=0)           # Shape (2B,)
            cond_combined = torch.cat([cond, cond_uncond], dim=0) # Shape (2B, D)

            # 3. Call model once
            #    IMPORTANT ASSUMPTION: model directly predicts noise 'eps'
            if self.model_mean_type != 'eps':
                # If model predicts x_0 or x_{t-1}, need to derive eps first
                # For now, raise error if assumption is violated
                raise NotImplementedError(f"CFG implementation currently assumes model_mean_type='eps', but got '{self.model_mean_type}'")
            model_output_combined = model(x_combined, t_combined, cond=cond_combined) # Shape (2B, ...) should be eps prediction; DiT forward

            # 4. Separate conditional and unconditional outputs
            cond_eps, uncond_eps = model_output_combined.chunk(2, dim=0) # Each shape (B, ...)

            # 5. Calculate guided noise prediction
            # Formula: eps = uncond_eps + scale * (cond_eps - uncond_eps)
            guided_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps) ### grad(log(p(y|x)))

            ### Calculate the conditional new mean
            ### Variance Role:
            # Acts as an adaptive step size
            # Larger variance → larger updates possible
            # Smaller variance → more conservative updates
            model_mean = model_mean + model_variance * guided_eps

        else: # No guidance (cond is None or cfg_scale <= 1.0)
            # Use the original p_mean_variance function to get predictions
            # Pass the original condition (might be None)
            model_mean, _, model_log_variance, pred_x0 = self.p_mean_variance(
                model, x, t, clip_denoised=clip_denoised, cond=cond, return_pred_x0=True
            )
        # --- End CFG Logic ---


        # --- Sampling Step (common to both cases) ---
        noise = noise_fn(x.shape, dtype=x.dtype).to(x.device)

        shape = [x.shape[0]] + [1] * (x.ndim - 1)
        nonzero_mask = (1 - (t == 0).type(torch.float32)).view(*shape).to(x.device) # mask for t=0

        # Calculate the sample using the potentially guided mean and log_variance
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

        # Optional: Clamp the final sample
        # sample = torch.clamp(sample, min=-1, max=1)

        # self.monitor.wandb_log({f'k60_noise_norm/{batch}': ((torch.exp(0.5 * log_var) * noise).norm().item())})
        # self.monitor.wandb_log({f'k60_noise_image/{batch}': wandb.Image(torch.exp(0.5 * log_var)* noise)})

        return (sample, pred_x0) if return_pred_x0 else sample

    @torch.no_grad()
    # def p_sample_loop(self, model, shape, noise_fn=torch.randn, lab=None):
    def p_sample_loop(self, model, shape, noise_fn=torch.randn, cond=None, cfg_scale=1.0):


        device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
        # shape[0] = lab.shape[0]
        img = noise_fn(shape).to(device)

        for i in reversed(range(self.num_timesteps)):
            img = self.p_sample(
                model,
                img,
                torch.full((shape[0],), i, dtype=torch.int64).to(device),
                noise_fn=noise_fn,
                cond=cond,
                cfg_scale=cfg_scale,
                return_pred_x0=False,
                # lab=lab,
            )

        return img

    @torch.no_grad()
    # def p_sample_loop_progressive(self, model, shape, device, cond=None, noise_fn=torch.randn, include_x0_pred_freq=50):
    def p_sample_loop_progressive(self, model, shape, device, cond=None, cfg_scale=1.0, noise_fn=torch.randn, include_x0_pred_freq=50):

        img = noise_fn(shape, dtype=torch.float32).to(device)
        num_recorded_x0_pred = self.num_timesteps // include_x0_pred_freq
        x0_preds_ = torch.zeros((shape[0], num_recorded_x0_pred, *shape[1:]), dtype=torch.float32).to(device)

        for i in reversed(range(self.num_timesteps)):
            ### revrse process: p(x_{t-1}|x_t) ≈ predict and denoise
            # img, pred_x0 = self.p_sample(model=model,
            #                              x=img,
            #                              t=torch.full((shape[0],), i, dtype=torch.int64).to(device),
            #                              cond=cond,
            #                              noise_fn=noise_fn,
            #                              return_pred_x0=True,
            #                             #  lab=cond,
            #                              )
            sample_output, pred_x0 = self.p_sample(model=model,
                                                 x=img,
                                                 t=torch.full((shape[0],), i, dtype=torch.int64).to(device),
                                                 noise_fn=noise_fn,
                                                 cond=cond,
                                                 cfg_scale=cfg_scale,
                                                 return_pred_x0=True)
            img = sample_output # Update img with the sampled output


            # Keep track of prediction of x0
            if i % include_x0_pred_freq == 0 or i == self.num_timesteps - 1: # Record more frequently or adjust logic
                idx = i // include_x0_pred_freq
                if 0 <= idx < num_recorded_x0_pred:
                     x0_preds_[:, idx, ...] = pred_x0

            # Original logic for inserting - might be less intuitive if include_x0_pred_freq is not 1
            # insert_mask = (torch.tensor(i // include_x0_pred_freq, device=device) == torch.arange(num_recorded_x0_pred, dtype=torch.int32, device=device))
            # insert_mask = insert_mask.to(torch.float32).view(1, num_recorded_x0_pred, *([1] * (len(shape) -1 ))) # Adjusted dimension calculation
            # x0_preds_ = insert_mask * pred_x0[:, None, ...] + (1. - insert_mask) * x0_preds_


        return img, x0_preds_

    def p_sample_loop_progressive_simple(self, model, shape, device, cond=None, cfg_scale=1.0, noise_fn=torch.randn,
                                         include_x0_pred_freq=50,input_pa=None,exp_step=None):

        # import pdb; pdb.set_trace()
        # sample_lab = torch.tensor([i%10 w i in range(shape[0])]).long().to(device)
        img = noise_fn(shape, dtype=torch.float32).to(device)
        if input_pa is not None:
            img = input_pa.repeat(shape[0],shape[1],1).to(device)
        # num_recorded_x0_pred = self.num_timesteps // include_x0_pred_freq # Not used here for history
        # x0_preds_            = torch.zeros((shape[0], num_recorded_x0_pred, *shape[1:]), dtype=torch.float32).to(device) # Not used here

        history = []
        if exp_step is not None:
            step = exp_step
        else:
            step = self.num_timesteps
        for i in reversed(range(step)):
            # import pdb;pdb.set_trace()

            # import pdb; pdb.set_trace()
            # Sample p(x_{t-1} | x_t) as usual
            # img, pred_x0 = self.p_sample(model=model,
            #                              x=img,
            #                              t=torch.full((shape[0],), i, dtype=torch.int64).to(device),
            #                              cond=cond,
            #                              noise_fn=noise_fn,
            #                              return_pred_x0=True,
            #                             #  lab=cond,
            #                             #  lab=cond,
            #                              )
            sample_output, pred_x0 = self.p_sample(model=model,
                                                   x=img,
                                                   t=torch.full((shape[0],), i, dtype=torch.int64).to(device),
                                                   noise_fn=noise_fn,
                                                   cond=cond,
                                                   cfg_scale=cfg_scale,
                                                   return_pred_x0=True)
            img = sample_output # Update img

            history.append(img.detach().cpu())
        return img, history

    # === Log likelihood calculation ===

    def _vb_terms_bpd(self, model, x_0, x_t, t, clip_denoised, return_pred_x0, cond=None):

        batch_size = t.shape[0]
        true_mean, _, true_log_variance_clipped    = self.q_posterior_mean_variance(x_0=x_0,
                                                                                    x_t=x_t,
                                                                                    t=t)
        model_mean, _, model_log_variance, pred_x0 = self.p_mean_variance(model,
                                                                          x=x_t,
                                                                          t=t,
                                                                          clip_denoised=clip_denoised,
                                                                          return_pred_x0=True,
                                                                          cond=cond,
                                                                          )

        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl = torch.mean(kl.view(batch_size, -1), dim=1) / np.log(2.)

        decoder_nll = -discretized_gaussian_log_likelihood(x_0, means=model_mean, log_scales=0.5 * model_log_variance)
        decoder_nll = torch.mean(decoder_nll.view(batch_size, -1), dim=1) / np.log(2.)

        # At the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where(t == 0, decoder_nll, kl)

        return (output, pred_x0) if return_pred_x0 else output


