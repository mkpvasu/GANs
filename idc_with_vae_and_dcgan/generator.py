import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, lat_emb_size, gen_fea_map_size, n_channels):
        super(Generator, self).__init__()
        self.latent_emb_size = lat_emb_size
        self.gen_feature_map_size = gen_fea_map_size
        self.n_channels = n_channels

        # generator nn
        self.block = nn.Sequential(
            # input is going to conv --> (gen_feature_map_size*8) * 4 * 4
            nn.ConvTranspose2d(self.latent_emb_size, self.gen_feature_map_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.gen_feature_map_size * 8),
            nn.ReLU(True),

            # (gen_feature_map_size*8) * 4 * 4 --> (gen_feature_map_size*4) * 8 * 8
            nn.ConvTranspose2d(self.gen_feature_map_size * 8, self.gen_feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.gen_feature_map_size * 4),
            nn.ReLU(True),

            # (gen_feature_map_size*4) * 8 * 8 --> (gen_feature_map_size*2) * 16 * 16
            nn.ConvTranspose2d(self.gen_feature_map_size * 4, self.gen_feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.gen_feature_map_size * 2),
            nn.ReLU(True),

            # (gen_feature_map_size * 2) * 16 * 16 --> (gen_feature_map_size) * 32 * 32
            nn.ConvTranspose2d(self.gen_feature_map_size * 2, self.gen_feature_map_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.gen_feature_map_size),
            nn.ReLU(True),

            # (gen_feature_map_size) * 32 * 32 --> (n_channels) * 64 * 64
            nn.ConvTranspose2d(self.gen_feature_map_size, n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, latent_embedding):
        return self.block(latent_embedding)
