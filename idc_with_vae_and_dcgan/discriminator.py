import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, dis_fea_map_size, n_channels):
        super(Discriminator, self).__init__()
        self.dis_feature_map_size = dis_fea_map_size
        self.n_channels = n_channels

        # discriminator nn
        self.block = nn.Sequential(
            # n_channels * 64 * 64 --> (dis_feature_map_size) * 32 * 32
            nn.Conv2d(self.n_channels, self.dis_feature_map_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (dis_feature_map_size) * 32 * 32 --> (dis_feature_map_size*2) * 16 * 16
            nn.Conv2d(self.dis_feature_map_size, self.dis_feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dis_feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (dis_feature_map_size*2) * 16 * 16 --> (dis_feature_map_size*4) * 8 * 8
            nn.Conv2d(self.dis_feature_map_size * 2, self.dis_feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dis_feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (dis_feature_map_size*4) * 8 * 8 --> (dis_feature_map_size*8) * 4 * 4
            nn.Conv2d(self.dis_feature_map_size * 4, self.dis_feature_map_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dis_feature_map_size * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # (dis_feature_map_size*8) * 4 * 4 --> discriminator_prediction_output
            nn.Conv2d(self.dis_feature_map_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, image):
        return self.block(image)
