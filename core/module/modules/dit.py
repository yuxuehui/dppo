# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_1d_sincos_pos_embed_from_grid(embed_dim, max_len=1000):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    pos = np.arange(max_len, dtype=np.float32)

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class FeatureReducer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureReducer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class DiT(nn.Module):
    def __init__(
            self,
            in_channel=1,
            patch_size=32,
            depth=2,
            nhead=4,
            hidden_size=32,
            mlp_ratio=4,
            condition_num=0,
            class_dropout_prob=0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.condition_num = condition_num
        self.class_dropout_prob = class_dropout_prob

        self.patch_embedder = PatchEmbed(patch_size=self.patch_size, embed_dim=self.hidden_size)
        self.feature_reducer = FeatureReducer(512, self.hidden_size)
        if self.condition_num != 0:
            print(self.condition_num)
            self.y_embedder = LabelEmbedder(condition_num, hidden_size, class_dropout_prob)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1000, hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, nhead, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.patch_size, in_channel)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1])
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))


        # # Initialize patch_embedding embedding MLP:
        nn.init.normal_(self.patch_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.patch_embedder.mlp[2].weight, std=0.02)

        # Initialize label embedding table:
        if self.condition_num != 0:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def token_drop(self, embeddings, force_drop_ids=None):
        """
        Drops embeddings to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(embeddings.shape[0], device=embeddings.device) < self.class_dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        embeddings = torch.where(drop_ids.unsqueeze(1), torch.zeros_like(embeddings), embeddings)
        return embeddings

    def forward(self, input, time, cond=None):
        input_shape = input.shape
        if len(input.size()) > 2:
            input = input.view(input.size(0), -1)
        input_view_shape = input.shape
        if input_view_shape[1] % self.patch_size != 0:
            input_pad = torch.cat(
                [
                    input,
                    torch.zeros(input_view_shape[0], self.patch_size - input_view_shape[1] % self.patch_size).to(
                        input.device
                    ),
                ],
                dim=1
            )
            input = input_pad.reshape(input_view_shape[0], -1, self.patch_size)
        else:
            input = input.reshape(input_view_shape[0], -1, self.patch_size)
        x = self.patch_embedder(input) + self.pos_embed[:, :input.size(1)]
        time_emb = self.t_embedder(time)
        if cond != None and self.condition_num != 0:
            cond = self.y_embedder(cond, self.training)
            c = time_emb + cond
        elif cond != None:
            # reduced_features = self.feature_reducer(cond)
            # #print("Reduced features shape:", reduced_features.shape)  # 应该是 (10, 64)
            # pooled_features = torch.mean(reduced_features, dim=1)
            # #print("Pooled features shape:", pooled_features.shape)  # 应该是 (64)
            # cond = pooled_features


            if self.training:
                cond = self.token_drop(cond)
            c = time_emb + cond
        else:
            c = time_emb
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)

        out = x.reshape(input_view_shape[0], -1)
        out = out[:, : input_view_shape[1]]
        out = out.reshape(input_shape)
        return out

class PatchEmbed(nn.Module):

    def __init__(self, patch_size=4, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.mlp = nn.Sequential(
            nn.Linear(patch_size, embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=True),
        )

    def forward(self, x):
        x = self.mlp(x)
        #x = self.norm(x)
        return x