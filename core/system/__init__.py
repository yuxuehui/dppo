from .base import *
from .ddpm import *
from .multitask_ae_ddpm import MultitaskAE_DDPM
from .multitask_vqvae_ddpm import MultitaskVQVAE_DDPM
from .multitask_episode_ddpm import MultitaskEpisode_DDPM

systems = {
    'ddpm': DDPM,
    'multitask_ae_ddpm': MultitaskAE_DDPM,
    'multitask_vqvae_ddpm': MultitaskVQVAE_DDPM,
    'multitask_episode_ddpm': MultitaskEpisode_DDPM
}