import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset,random_split
from torchvision import transforms
import random
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger


class AtariDataset(Dataset):
    def __init__(self, data_files, transform=None):
        self.data = []
        self.transform = transform
        for file in data_files:
            with open(file, 'rb') as f:
                self.data.extend(pickle.load(f))

        # 将数据展平为 [200*100, 50, (state, action, reward, next_state)]
        self.data = [episode for sublist in self.data for episode in sublist if len(episode) >= 10]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        episode = self.data[idx]  # [50, (state, action, reward, next_state)]

        # 随机选择一个连续的大小为 10 的子列表
        start_idx = random.randint(0, len(episode) - 10)  # 40 是 50 - 10
        sub_episode = episode[start_idx:start_idx + 10]

        # 提取 state 并构建 [10, state]
        #states = np.array([step[0] for step in sub_episode])  # [10, 84, 84, 4]
        states = np.array([step for step in sub_episode])  # [10, 84, 84, 4]

        if self.transform:
            states = np.array([self.transform(state) for state in states])

        states = states / 255.0

        # 转换为 PyTorch 张量并确保 dtype 一致
        states = torch.tensor(states, dtype=torch.float32)

        return states

class Encoder(nn.Module):
    def __init__(self, token_dim, hidden_dim, num_heads, num_layers, patch_size=21):
        super(Encoder, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (84 // patch_size) * (84 // patch_size)  # Assuming input image size is 84x84
        self.token_dim = token_dim

        self.image_mlp = nn.Sequential(
            nn.Linear(patch_size * patch_size * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, token_dim)
        )

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches * 10, token_dim))

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=token_dim, nhead=num_heads),
            num_layers=num_layers
        )

    def forward(self, img_states):
        batch_size, seq_len, _, _, _ = img_states.size()
        img_states = img_states.view(batch_size * seq_len, 84, 84, 4)  # 展平成(batch_size * seq_len, 84, 84, 4)

        # 分割图像为patches
        patches = img_states.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size * seq_len, -1, self.patch_size * self.patch_size * 4)  # (batch_size * seq_len, num_patches, patch_size * patch_size * 4)

        # 通过MLP转换为tokens
        tokens = self.image_mlp(patches)  # (batch_size * seq_len, num_patches, token_dim)

        # 恢复batch和seq_len维度
        tokens = tokens.view(batch_size, seq_len * self.num_patches, self.token_dim)  # (batch_size, seq_len * num_patches, token_dim)

        # 添加位置嵌入
        tokens += self.position_embedding[:, :seq_len * self.num_patches, :]

        # Transformer编码
        transformer_output = self.transformer_encoder(tokens)  # (batch_size, seq_len * num_patches, token_dim)

        return transformer_output


class Decoder(nn.Module):
    def __init__(self, token_dim, hidden_dim, num_heads, num_layers, patch_size=21):
        super(Decoder, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (84 // patch_size) * (84 // patch_size)  # Assuming input image size is 84x84
        self.token_dim = token_dim

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches * 10, token_dim))

        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=token_dim, nhead=num_heads),
            num_layers=num_layers
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, patch_size * patch_size * 4)
        )

    def forward(self, encoded_tokens, memory):
        batch_size, seq_len_patches, _ = encoded_tokens.size()

        # 添加位置嵌入
        encoded_tokens += self.position_embedding[:, :seq_len_patches, :]

        # Transformer解码
        decoded_tokens = self.transformer_decoder(encoded_tokens, memory)  # (batch_size, seq_len * num_patches, token_dim)

        # 通过MLP转换为patches
        patches = self.output_mlp(decoded_tokens)  # (batch_size, seq_len * num_patches, patch_size * patch_size * 4)

        # 恢复图像形状
        patches = patches.view(batch_size, seq_len_patches // self.num_patches, self.num_patches, self.patch_size, self.patch_size, 4)
        patches = patches.permute(0, 1, 3, 4, 2, 5).contiguous()
        img_states = patches.view(batch_size, seq_len_patches // self.num_patches, 84, 84, 4)

        return img_states


class VAE(pl.LightningModule):
    def __init__(self, img_input_dim, token_dim, hidden_dim, latent_dim, num_heads, num_layers, img_output_dim, learning_rate=1e-4, patch_size=21, only_ae=False):
        super(VAE, self).__init__()
        self.encoder = Encoder(token_dim, hidden_dim, num_heads, num_layers, patch_size=patch_size)
        self.fc_mu = nn.Linear(token_dim, latent_dim)
        self.fc_logvar = nn.Linear(token_dim, latent_dim)
        self.fc_z = nn.Linear(latent_dim, token_dim)
        self.decoder = Decoder(token_dim, hidden_dim, num_heads, num_layers, patch_size=patch_size)
        self.learning_rate = learning_rate
        self.only_ae = only_ae

    def encode(self, img_states):
        h = self.encoder(img_states)
        h = h.mean(dim=1)  # 对序列维度取平均，得到(batch_size, token_dim)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, memory):
        z = self.fc_z(z).unsqueeze(1).repeat(1, memory.size(1), 1)  # 重新调整形状为(batch_size, seq_len, token_dim)
        return self.decoder(z, memory)

    def forward(self, img_states):
        mu, logvar = self.encode(img_states)
        z = self.reparameterize(mu, logvar)
        memory = self.encoder(img_states)  # 获取编码器的输出作为解码器的memory
        img_output = self.decode(z, memory)
        return img_output, mu, logvar

    def training_step(self, batch, batch_idx):
        img_states = batch  # 现在 batch 只包含一个元素
        img_output, mu, logvar = self(img_states)
        recon_loss = F.mse_loss(img_output, img_states, reduction='mean')

        if self.only_ae:
            loss = recon_loss
        else:
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            self.log('kld_loss', kld_loss, prog_bar=True)
            loss = recon_loss + kld_loss

        self.log('recon_loss', recon_loss, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img_states = batch  # 现在 batch 只包含一个元素
        img_output, mu, logvar = self(img_states)
        recon_loss = F.mse_loss(img_output, img_states, reduction='mean')

        if self.only_ae:
            loss = recon_loss
        else:
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            self.log('val_kld_loss', kld_loss, prog_bar=True)
            loss = recon_loss + kld_loss

        self.log('val_recon_loss', recon_loss, prog_bar=True)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

def split_dataset(dataset, train_ratio=0.9):
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset


# 数据加载和预处理
#data_files = ["/mnt/kaiwu-group-x3/jiateliu/policy-diffusion/param_data/Atari_zoo/Atari-Assault/200/episodes_data_test.pkl"]
data_files = ["/mnt/kaiwu-group-x3/jiateliu/policy-diffusion/param_data/Atari_zoo/Atari-Alien/200/episodes_data2.pkl",
              "/mnt/kaiwu-group-x3/jiateliu/policy-diffusion/param_data/Atari_zoo/Atari-Amidar/200/episodes_data2.pkl",
              "/mnt/kaiwu-group-x3/jiateliu/policy-diffusion/param_data/Atari_zoo/Atari-Assault/200/episodes_data2.pkl",
              "/mnt/kaiwu-group-x3/jiateliu/policy-diffusion/param_data/Atari_zoo/Atari-Asteroids/200/episodes_data2.pkl",
              "/mnt/kaiwu-group-x3/jiateliu/policy-diffusion/param_data/Atari_zoo/Atari-BattleZone/200/episodes_data2.pkl"
              ]

val_data_files = ["/mnt/kaiwu-group-x3/jiateliu/policy-diffusion/param_data/Atari_zoo/Atari-BeamRider/200/episodes_data2.pkl"

                  ]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = AtariDataset(data_files)
if len(val_data_files) > 0:
    val_dataset = AtariDataset(val_data_files)
    train_dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)
else:
    train_dataset, val_dataset = split_dataset(dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# 模型参数
img_input_dim = 84 * 84 * 4  # Atari图像大小
token_dim = 32
hidden_dim = 256
latent_dim = 64
num_heads = 4
num_layers = 2
img_output_dim = 84 * 84 * 4  # Atari图像大小
patch_size = 84

save_dir = '/mnt/kaiwu-group-x3/jiateliu/policy-diffusion/episode_outputs/5-1/'
#save_dir = '/mnt/kaiwu-group-x3/jiateliu/policy-diffusion/episode_outputs/Atari-Assault/'

# 初始化模型
model = VAE(img_input_dim, token_dim, hidden_dim, latent_dim, num_heads, num_layers, img_output_dim, patch_size=patch_size, only_ae=True)



logger = TensorBoardLogger(
    save_dir=save_dir,
    name='log'  # 子目录名称
)

# 模型检查点回调
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=save_dir,
    filename='vae-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min',
)

# 自定义进度条回调
class CustomProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.set_description('Training')
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description('Validating')
        return bar

# 训练
trainer = pl.Trainer(
    max_epochs=500,
    callbacks=[checkpoint_callback],
    check_val_every_n_epoch=2,
    logger=logger
)
trainer.fit(model, train_dataloader, val_dataloader)
