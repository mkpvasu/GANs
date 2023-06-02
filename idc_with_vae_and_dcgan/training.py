import os
import glob
import random
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.utils
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from generator import Generator
from discriminator import Discriminator
from vae_encoder import VanillaVAEEncoder

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


class CustomImageDataset(Dataset):
    def __init__(self, path, pattern, transform=None):
        self.file_list = glob.glob(os.path.join(path, pattern))
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        output = {}
        data = np.load(self.file_list[idx])
        delta_vmap = torch.tensor(data['delta_vmap'], dtype=torch.float)
        delta_vmap = torch.reshape(delta_vmap, (1, 64, 64))
        output["delta_vmap"] = delta_vmap

        dI = torch.tensor(data["dI"], dtype=torch.float)
        output["dI"] = dI

        dmap = data['dmap']
        # for idx, row in enumerate(dmap):
        #     for jdx, pixel in enumerate(row):
        #         if pixel > 300:
        #             dmap[idx][jdx] = 0

        dmap = torch.tensor(dmap, dtype=torch.float)
        dmap = torch.reshape(dmap, (1, 64, 64))
        nmap = torch.tensor(data['nmap'], dtype=torch.float)
        nmap = nmap.permute(2, 0, 1)
        combined_map = torch.cat((dmap, nmap), dim=0)
        # combined_map = dmap

        if self.transform:
            combined_map = self.transform(combined_map)
        output["combined_map"] = combined_map

        return output


class Data:
    def __init__(self):
        self.dataroot = os.path.join(os.getcwd(), "data", "dcgan_data")
        self.pattern = "d_*.npz"
        self.image_size = 64
        self.batch_size = 16
        self.shuffle = True
        self.num_workers = 2
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((64, 64), antialias=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5]),  # Add data normalization
        ])

    def dataset_prep(self):
        dataset = CustomImageDataset(path=self.dataroot, pattern=self.pattern, transform=self.transform)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                num_workers=self.num_workers)
        return dataloader


class Training:
    def __init__(self, n_epochs: int = 10):
        # set random seed for reproducibility
        self.manualSeed = 1000
        # manualSeed = random.randint(1, 10000) # use if you want new results
        random.seed(self.manualSeed)
        torch.manual_seed(self.manualSeed)
        # in_channels
        self.in_channels = 4
        # rgb image
        self.n_channels = 1
        # latent space input
        self.latent_dims = 64
        # total latent dims including dI values
        self.total_latent_dims = 448
        # size of generator feature map
        self.gen_feature_map_size = 64
        # size of discriminator feature map
        self.dis_feature_map_size = 64
        # labels
        self.labels = {"real": 1.0, "fake": 0.0}
        # loss function - binary cross entropy loss
        self.criterion = nn.BCELoss()
        # loss function for VAE
        self.vae_criterion = nn.KLDivLoss(reduction="batchmean")
        # vae optmizer
        self.vae_optimizer = optim.Adam
        # vae_loss_hyperparameter
        self.vae_loss_param = 0.01
        # generator optimizer
        self.gen_optimizer = optim.Adam
        # discriminator optimizer
        self.dis_optimizer = optim.Adam
        # vae encoder learning rate
        self.vae_lr = 0.001
        # generator learning rate
        self.gen_lr = 0.005
        # discriminator learning rate
        self.dis_lr = 0.005
        # optimizer beta
        self.betas = (0.5, 0.999)
        # number of epochs
        self.n_epochs = n_epochs
        # device to run model
        self.device = device

    # Suggested by DCGAN paper to initialize the initial generator image with random noises having mean = 0 and std =
    # 0.02
    def weight_initialization(self, gen):
        class_name = gen.__class__.__name__
        if class_name.find("Conv") != -1:
            nn.init.normal_(gen.weight.data, 0.0, 0.02)
        elif class_name.find("BatchNorm") != -1:
            nn.init.normal_(gen.weight.data, 1.0, 0.02)
            nn.init.constant_(gen.bias.data, 0)

    def training(self):
        # initialize vae encoder
        vae_encoder = VanillaVAEEncoder(in_channels=self.in_channels, latent_dim=self.latent_dims)
        # initialize generator
        generator = Generator(self.total_latent_dims, self.gen_feature_map_size,
                              self.n_channels).to(device=self.device)
        generator.apply(self.weight_initialization)
        # initialize discriminator
        discriminator = Discriminator(self.dis_feature_map_size, self.n_channels).to(self.device)
        discriminator.apply(self.weight_initialization)

        fixed_noise = torch.randn(64, self.total_latent_dims, 1, 1, device=self.device)
        criterion = self.criterion
        vae_criterion = self.vae_criterion
        vae_optimizer = self.vae_optimizer(vae_encoder.parameters(), lr=self.vae_lr, betas=self.betas)
        gen_optimizer = self.gen_optimizer(generator.parameters(), lr=self.gen_lr, betas=self.betas)
        dis_optimizer = self.dis_optimizer(discriminator.parameters(), lr=self.dis_lr, betas=self.betas)
        dataloader = Data().dataset_prep()

        # Lists to keep track of progress
        img_list = []
        gen_losses = []
        dis_losses = []
        iters = 0

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(self.n_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(dataloader, 0):

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # Train with all-real batch
                discriminator.zero_grad()
                # Format batch
                real_cpu = data["delta_vmap"].to(device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), self.labels["real"], dtype=torch.float, device=device)
                # Forward pass real batch through D
                output = discriminator(real_cpu).view(-1)
                # Calculate loss on all-real batch
                dis_error_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                dis_error_real.backward()
                D_x = output.mean().item()

                # Train with all-fake batch
                # Generate batch of latent vectors
                combined_map = data["combined_map"]
                vae_latent_embedding = vae_encoder.forward(combined_map).to(self.device)
                vae_latent_embedding = nn.functional.normalize(vae_latent_embedding)
                di = data["dI"].to(self.device)
                vae_latent_embedding_with_di = torch.cat((vae_latent_embedding, di), dim=1).to(self.device)
                vae_latent_embed_vector = torch.reshape(vae_latent_embedding_with_di,
                                                        (b_size, self.total_latent_dims, 1, 1))

                # Generate fake image batch with G
                fake = generator(vae_latent_embed_vector)
                label.fill_(self.labels["fake"])
                # Classify all fake batch with D
                output = discriminator(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                dis_error_fake = criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                dis_error_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                dis_error = dis_error_real + dis_error_fake
                # Update D
                dis_optimizer.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                generator.zero_grad()
                label.fill_(self.labels["real"])  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = discriminator(fake).view(-1)
                # Calculate G's loss based on this output
                gen_error = criterion(output, label)
                # Update VAE Encoder to generate better latent vectors
                delta_vmaps = data["delta_vmap"].to(self.device)
                # vae error between fake and delta_maps
                softmax_fake = nn.functional.softmax(fake, dim=2)
                softmax_d_vmaps = nn.functional.softmax(delta_vmaps, dim=2)
                vae_error = vae_criterion(softmax_fake, softmax_d_vmaps)
                # total error
                total_error = gen_error + (- self.vae_loss_param * vae_error)
                # Calculate gradients for G
                total_error.backward()
                D_G_z2 = output.mean().item()
                # Update G
                gen_optimizer.step()

                # vae_error.backward(retain_graph=True)
                # Update VAE
                vae_optimizer.step()

                # Output training stats
                if i % 50 == 0:
                    # Output training stats
                    if i % 50 == 0:
                        print(f"[{epoch}/{self.n_epochs}][{i}/{len(dataloader)}]\tVAE Loss: {vae_error.item():.4f}"
                              f"\tLoss_D: {dis_error.item():.4f} \tLoss_G: {gen_error.item():.4f}\tD(x): {D_x:.4f}"
                              f"\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}")

                # Save Losses for plotting later
                gen_losses.append(gen_error.item())
                dis_losses.append(dis_error.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == self.n_epochs - 1) and (i == len(dataloader) - 1)):
                    with torch.no_grad():
                        fake = generator(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1

        return img_list, gen_losses, dis_losses

    def execute(self):
        return self.training()


class Visualize_Model:
    def __init__(self):
        self.gen_imgs, self.gen_losses, self.dis_losses = Training(n_epochs=10).execute()

    def save_gen_images(self):
        path = os.path.join(os.getcwd(), "generated_images")
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        for i in range(len(self.gen_imgs)):
            torchvision.utils.save_image(tensor=self.gen_imgs[i, :, :, :], fp=os.path.join(path, f"{i:02d}.jpg"))

    def gen_dis_loss_training(self):
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.gen_losses, label="G")
        plt.plot(self.dis_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def gen_output(self):
        fig = plt.figure(figsize=(8, 8))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in self.gen_imgs]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

        HTML(ani.to_jshtml())
        plt.show()
        ani.save("idc_vae_dcgan_animation.gif", writer="pillow", fps=1)
        # self.save_gen_images()


def main():
    Visualize_Model().gen_output()


if __name__ == "__main__":
    main()
