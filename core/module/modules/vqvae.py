import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import TransformerEncoder, TransformerDecoder
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer
import random
from vector_quantize_pytorch import ResidualVQ


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, inputs):
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embeddings.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embeddings.weight).view_as(inputs)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity, encodings


class VQVAETransformer(nn.Module):
    def __init__(self, d_model, nhead, depth, num_embeddings, commitment_cost, condition_num=0, dim_feedforward=2048,
                 dropout=0.1, patch_size=64):
        super(VQVAETransformer, self).__init__()
        self.patch_size = patch_size
        self.condition_num = condition_num
        if self.condition_num != 0:
            self.y_embedder = LabelEmbedder(condition_num, d_model, 0)

        self.embed_layer = nn.Linear(self.patch_size, d_model)
        self.decode_layer = nn.Linear(d_model, self.patch_size)

        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = TransformerEncoder(encoder_layers, depth)

        decoder_layers = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(decoder_layers, depth)

        #self.vq_layer = VectorQuantizer(num_embeddings, d_model, commitment_cost)
        self.vq_layer = ResidualVQ(
            dim=d_model,
            num_quantizers=8,  # You can adjust this based on your needs
            codebook_size=num_embeddings,
            stochastic_sample_codes=True,
            sample_codebook_temp=0.1,
            shared_codebook=True
        )

    def encode(self, input, condition=None):
        bs, self.input_dim = input.shape
        pad_size = self.patch_size - (self.input_dim % self.patch_size)
        input_pad = torch.cat([input, torch.zeros(bs, pad_size).to(input.device)], dim=1)

        input_seq = input_pad.reshape(bs, -1, self.patch_size)

        embeddings = self.embed_layer(input_seq)
        if self.condition_num != 0 and condition is not None:
            condition = self.y_embedder(condition)
        if condition is not None:
            embeddings = torch.cat([condition.unsqueeze(1), embeddings], dim=1)

        latent = self.encoder(embeddings)
        #quantized, vq_loss, perplexity, _ = self.vq_layer(latent)
        quantized, indices, commit_loss = self.vq_layer(latent)
        #perplexity = torch.exp(-torch.sum(indices.float() * torch.log(indices.float() + 1e-10)) / indices.numel())
        commit_loss = commit_loss.mean()
        return quantized, commit_loss

    def decode(self, quantized, condition=None):
        bs = quantized.shape[0]
        embeddings = quantized
        if self.condition_num != 0 and condition is not None:
            condition = self.y_embedder(condition)
        if condition is not None:
            embeddings = torch.cat([condition.unsqueeze(1), quantized], dim=1)

        output = self.decoder(embeddings, embeddings)
        output = self.decode_layer(output)

        output = output.reshape(bs, -1)
        output = output[:, : self.input_dim]
        return output

    def forward(self, input):
        quantized, commit_loss = self.encode(input)
        #vq_loss = vq_loss.sum()
        output = self.decode(quantized)
        return output, commit_loss



