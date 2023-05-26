import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from generator import Generator
from discriminator import Discriminator

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


class Data:
    def __init__(self):
        self.dataroot = os.path.join(os.getcwd(), "celeba")
        self.image_size = 64
        self.batch_size = 16
        self.shuffle = True
        self.num_workers = 2
        self.device = device

    def dataset_prep(self):
        dataset = datasets.ImageFolder(root=self.dataroot,
                                       transform=transforms.Compose([
                                           transforms.Resize(self.image_size),
                                           transforms.CenterCrop(self.image_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                       ]))
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                num_workers=self.num_workers)

        # batch = next(iter(dataloader))
        # plt.figure(figsize=(8, 8))
        # plt.axis("off")
        # plt.title("Training Images")
        # plt.imshow(np.transpose(vutils.make_grid(batch[0].to(self.device)[:64], padding=2,
        #                                          normalize=True).cpu(), (1, 2, 0)))
        # plt.show()
        return dataloader


class Training:
    def __init__(self):
        # set random seed for reproducibility
        self.manualSeed = 1000
        # manualSeed = random.randint(1, 10000) # use if you want new results
        random.seed(self.manualSeed)
        torch.manual_seed(self.manualSeed)
        # rgb image
        self.n_channels = 3
        # latent space input
        self.latent_embedding_size = 128
        # size of generator feature map
        self.gen_feature_map_size = 64
        # size of discriminator feature map
        self.dis_feature_map_size = 64
        # labels
        self.labels = {"real": 1.0, "fake": 0.0}
        # loss function - binary cross entropy loss
        self.criterion = nn.BCELoss()
        # generator optimizer
        self.gen_optimizer = optim.Adam
        # discriminator optimizer
        self.dis_optimizer = optim.Adam
        # lr
        self.lr = 0.0002
        # optimizer beta
        self.betas = (0.5, 0.999)
        # number of epochs
        self.n_epochs = 3
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
        # initialize generator
        generator = Generator(self.latent_embedding_size, self.gen_feature_map_size,
                              self.n_channels).to(device=self.device)
        generator.apply(self.weight_initialization)
        # initialize discriminator
        discriminator = Discriminator(self.dis_feature_map_size, self.n_channels).to(self.device)
        discriminator.apply(self.weight_initialization)

        fixed_noise = torch.randn(64, self.latent_embedding_size, 1, 1, device=self.device)
        criterion = self.criterion
        gen_optimizer = self.gen_optimizer(generator.parameters(), lr=self.lr, betas=self.betas)
        dis_optimizer = self.dis_optimizer(discriminator.parameters(), lr=self.lr, betas=self.betas)
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
                ## Train with all-real batch
                discriminator.zero_grad()
                # Format batch
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), self.labels["real"], dtype=torch.float, device=device)
                # Forward pass real batch through D
                output = discriminator(real_cpu).view(-1)
                # Calculate loss on all-real batch
                dis_error_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                dis_error_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.latent_embedding_size, 1, 1, device=device)
                # Generate fake image batch with G
                fake = generator(noise)
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
                # Calculate gradients for G
                gen_error.backward()
                D_G_z2 = output.mean().item()
                # Update G
                gen_optimizer.step()

                # Output training stats
                if i % 50 == 0:
                    # Output training stats
                    if i % 50 == 0:
                        print(f"[{epoch}/{self.n_epochs}][{i}/{len(dataloader)}]\tLoss_D: {dis_error.item():.4f} "
                              f"\tLoss_G: {gen_error.item():.4f}\tD(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}")

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
        self.gen_imgs, self.gen_losses, self.dis_losses = Training().execute()

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


def main():
    Visualize_Model().gen_output()


if __name__ == "__main__":
    main()
