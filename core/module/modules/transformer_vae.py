## Adopted from https://github.com/rosinality/denoising-diffusion-pytorch with some minor changes.

import math
import pdb
import random

import torch
from torch import nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        #x = x + self.pe[: x.size(0), :]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

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

    def forward(self, labels, train=False, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class TransformerEncoder(nn.Module):
    def __init__(self, nhead, dim_feedforward, num_layers, d_model, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pos_encoder = PositionalEncoding(d_model)

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, nhead, dim_feedforward, num_layers, d_model, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.pos_decoder = PositionalEncoding(d_model)

    def forward(self, tgt, memory):
        tgt = self.pos_decoder(tgt)
        output = self.transformer_decoder(tgt, memory)
        return output


class TransformerAE_old(nn.Module):
    def __init__(self, in_dim, d_model, nhead, nlayer, condition_num=0, dim_feedforward=2048, dropout=0.1, patch_size=64, input_noise_factor=0.001, latent_noise_factor=0.1, depth=1):
        super(TransformerAE_old, self).__init__()
        self.patch_size = patch_size
        self.input_noise_factor = input_noise_factor
        self.latent_noise_factor = latent_noise_factor
        if condition_num != 0:
            self.y_embedder = LabelEmbedder(condition_num, d_model, 0)
        self.embed_layer = nn.Linear(self.patch_size, d_model)
        self.decode_layer = nn.Linear(d_model, self.patch_size)
        self.encoder = TransformerEncoder(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, num_layers=nlayer, dropout=dropout)
        self.decoder = TransformerDecoder(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, num_layers=nlayer, dropout=dropout)

        # self.encoder_blocks = nn.ModuleList([
        #     TransformerEncoder(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, num_layers=nlayer, dropout=dropout)
        # ])
        # self.decoder_blocks = nn.ModuleList([
        #     TransformerDecoder(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, num_layers=nlayer, dropout=dropout)
        # ])

    def encode(self, input, condition=None):
        bs, self.input_dim = input.shape
        pad_size = self.patch_size - (self.input_dim % self.patch_size)
        # if condition is not None:
        #     input_pad = torch.cat([input, condition.reshape([input.shape[0], 1]).repeat(1, pad_size).to(input.device)], dim=1)
        # else:
        input_pad = torch.cat([input, torch.zeros(bs, pad_size).to(input.device)], dim=1)

        input_seq = input_pad.reshape(bs, -1, self.patch_size)
        input_seq = self.add_noise(input_seq, self.input_noise_factor)

        embeddings = self.embed_layer(input_seq)
        if condition is not None:
            condition = self.y_embedder(condition)
            embeddings = torch.cat([condition.unsqueeze(1), embeddings], dim=1)

        latent = self.encoder(embeddings)
        # for block in self.encoder_blocks:
        #     embeddings = block(embeddings)
        # latent = embeddings
        return latent

    def decode(self, latent, condition=None):
        bs = latent.shape[0]
        embeddings = self.add_noise(latent, self.latent_noise_factor)
        if condition is not None:
            condition = self.y_embedder(condition)
            embeddings = torch.cat([condition.unsqueeze(1), latent], dim=1)

        output = self.decoder(embeddings, embeddings)
        # output = embeddings
        # for block in self.decoder_blocks:
        #     output = block(output, embeddings)

        output = self.decode_layer(output)

        output = output.reshape(bs, -1)
        output = output[:, : self.input_dim]
        return output

    def forward(self, input, condition=None):
        latent = self.encode(input, condition)
        output = self.decode(latent, condition)
        return output

    def add_noise(self, x, noise_factor):
        if not isinstance(noise_factor, float):
            assert len(noise_factor) == 2
            noise_factor = random.uniform(noise_factor[0], noise_factor[1])
        return torch.randn_like(x) * noise_factor + x * (1 - noise_factor)

class TransformerAE(nn.Module):
    def __init__(self, in_dim, d_model, nhead, nlayer, condition_num=0, dim_feedforward=2048, dropout=0.1, patch_size=64, input_noise_factor=0.001, latent_noise_factor=0.1, depth=1):
        super(TransformerAE, self).__init__()
        self.patch_size = patch_size
        self.input_noise_factor = input_noise_factor
        self.latent_noise_factor = latent_noise_factor
        if condition_num != 0:
            self.y_embedder = LabelEmbedder(condition_num, d_model, 0)
        self.embed_layer = nn.Linear(self.patch_size, d_model)
        self.decode_layer = nn.Linear(d_model, self.patch_size)
        # self.encoder = TransformerEncoder(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, num_layers=nlayer, dropout=dropout)
        # self.decoder = TransformerDecoder(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, num_layers=nlayer, dropout=dropout)

        self.encoder_blocks = nn.ModuleList([
            TransformerEncoder(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, num_layers=nlayer, dropout=dropout) for _ in range(depth)
        ])
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoder(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, num_layers=nlayer, dropout=dropout) for _ in range(depth)
        ])

    def encode(self, input, condition=None):
        bs, self.input_dim = input.shape
        pad_size = self.patch_size - (self.input_dim % self.patch_size)
        # if condition is not None:
        #     input_pad = torch.cat([input, condition.reshape([input.shape[0], 1]).repeat(1, pad_size).to(input.device)], dim=1)
        # else:
        input_pad = torch.cat([input, torch.zeros(bs, pad_size).to(input.device)], dim=1)

        input_seq = input_pad.reshape(bs, -1, self.patch_size)
        input_seq = self.add_noise(input_seq, self.input_noise_factor)

        embeddings = self.embed_layer(input_seq)
        if condition is not None:
            condition = self.y_embedder(condition)
            embeddings = torch.cat([condition.unsqueeze(1), embeddings], dim=1)

        #latent = self.encoder(embeddings)
        for block in self.encoder_blocks:
            embeddings = block(embeddings)
        latent = embeddings
        return latent

    def decode(self, latent, condition=None):
        bs = latent.shape[0]
        embeddings = self.add_noise(latent, self.latent_noise_factor)
        if condition is not None:
            condition = self.y_embedder(condition)
            embeddings = torch.cat([condition.unsqueeze(1), latent], dim=1)

        #output = self.decoder(embeddings, embeddings)
        output = embeddings
        for block in self.decoder_blocks:
            output = block(output, embeddings)

        output = self.decode_layer(output)

        output = output.reshape(bs, -1)
        output = output[:, : self.input_dim]
        return output

    def forward(self, input, condition=None):
        latent = self.encode(input, condition)
        output = self.decode(latent, condition)
        return output

    def add_noise(self, x, noise_factor):
        if not isinstance(noise_factor, float):
            assert len(noise_factor) == 2
            noise_factor = random.uniform(noise_factor[0], noise_factor[1])
        return torch.randn_like(x) * noise_factor + x * (1 - noise_factor)