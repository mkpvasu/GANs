import torch
from torch import nn
import numpy as np


class VanillaVAEEncoder:
    def __init__(self,
                 latent_dim: int,
                 **kwargs) -> None:
        super(VanillaVAEEncoder, self).__init__()

        self.in_channels = 1
        self.out_channels = 1
        self.latent_dim = latent_dim
        self.hidden_dims = [32, 64, 128]
        self.encoder = self.encoder_block()

        # # Build Decoder
        # modules = []
        # self.decoder_input = nn.Linear(latent_dim, self.hidden_dims[-1] * 4)
        # self.hidden_dims.reverse()
        #
        # for i in range(len(self.hidden_dims) - 1):
        #     modules.append(
        #         nn.Sequential(
        #             nn.ConvTranspose2d(self.hidden_dims[i],
        #                                self.hidden_dims[i + 1],
        #                                kernel_size=3,
        #                                stride=2,
        #                                padding=1,
        #                                output_padding=1),
        #             nn.BatchNorm2d(self.hidden_dims[i + 1]),
        #             nn.LeakyReLU())
        #     )
        #
        # self.decoder = nn.Sequential(*modules)
        # self.final_layer = nn.Sequential(
        #                     nn.ConvTranspose2d(self.hidden_dims[-1],
        #                                        self.hidden_dims[-1],
        #                                        kernel_size=3,
        #                                        stride=2,
        #                                        padding=1,
        #                                        output_padding=1),
        #                     nn.BatchNorm2d(self.hidden_dims[-1]),
        #                     nn.LeakyReLU(),
        #                     nn.Conv2d(self.hidden_dims[-1], out_channels=self.out_channels,
        #                               kernel_size=3, padding=1),
        #                     nn.Sigmoid())

    def encoder_block(self):
        modules = []
        in_channels = self.in_channels
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        encoder = nn.Sequential(*modules)
        return encoder

    def calculate_mu_var(self, result, result_dim):
        fc_mu = nn.Linear(result_dim, self.latent_dim)
        fc_var = nn.Linear(result_dim, self.latent_dim)
        return fc_mu(result), fc_var(result)

    def encode(self, input: torch.tensor) -> list[torch.tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (torch.tensor) Input torch.tensor to encoder [N x C x H x W]
        :return: (torch.tensor) list of latent codes
        """
        encoder_output = self.encoder(input)
        result = torch.flatten(encoder_output, start_dim=1)
        result_dim = result.shape[1]

        # Split the result into mu and var components of the latent Gaussian distribution

        mu, log_var = self.calculate_mu_var(result, result_dim)
        return [mu, log_var]

    def reparameterize(self, mu: torch.tensor, logvar: torch.tensor) -> torch.tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (torch.tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (torch.tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (torch.tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def latent_embedding(self, input: torch.tensor, **kwargs) -> list[torch.tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return z


def main():

    for i in range(1):
        data = np.load(f".\\data\\vae_data\\d_{i}.npz")
        for key in data:
            print(key)

        # delta_vmap = data['delta_vmap']
        dmap = torch.from_numpy(data['dmap'])
        dmap = torch.reshape(dmap, (1, 64, 64))
        nmap = torch.from_numpy(data['nmap'])
        nmap = nmap.permute(2, 0, 1)
        img = torch.cat((nmap, dmap), dim=0)
        img = torch.reshape(img, (1, 4, 64, 64))
        z = VanillaVAEEncoder(latent_dim=16).latent_embedding(img)
        print(z)


if __name__ == "__main__":
    main()
