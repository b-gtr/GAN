import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import random

import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


# -----------------------------
#   Discriminator Definition
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self, input_channel, ndf=16):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            nn.Conv2d(input_channel, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*8, ndf*16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*16, ndf*32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*32, ndf*64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*64, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


# -----------------------------
#   Generator Definition
# -----------------------------
class Generator(nn.Module):
    def __init__(self, latent_space, ngf, input_channel):
        super(Generator,self).__init__()
        
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(latent_space, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, input_channel, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, inp):
        return self.generator(inp)


def image_dataset(IMAGE_SIZE, DATA_PATH, BATCH_SIZE):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.9, 1.0)), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2), 
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = torchvision.datasets.ImageFolder(
        root=DATA_PATH,
        transform=transform
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    return dataloader


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def main():
    # Hyperparameters, paths, etc.
    #DATA_PATH = r"C:\Users\\tiny_images"
    DATA_PATH = r"C:\Users\face_test"
    IMAGE_SIZE = 512
    NC = 1
    NZ = 100
    NGF = 64
    NDF = 8
    BATCH_SIZE = 64
    EPOCHS = 150
    LR = 2e-4
    BETA1 = 0.5
    BETA2 = 0.999
    NGPU = 0
    LATENT_DIM = 100

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("amples_gray", exist_ok=True)

    dataloader = image_dataset(
        IMAGE_SIZE=IMAGE_SIZE, 
        DATA_PATH=DATA_PATH, 
        BATCH_SIZE=BATCH_SIZE
    )

    device = torch.device("cuda:0" if (torch.cuda.is_available() and NGPU > 0) else "cpu")

    netG = Generator(LATENT_DIM, NGF, NC).to(DEVICE)
    netD = Discriminator(NC, NDF).to(DEVICE)

    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, BETA2))
    optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, BETA2))

    fixed_noise = torch.randn(BATCH_SIZE, LATENT_DIM, 1, 1, device=DEVICE)

    real_label = 1.
    fake_label = 0.

    # For tracking
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")

    # ------
    # Configure how many times to update G per D step
    num_gen_steps = 3  # Increase or decrease as you like
    # ------

    for epoch in range(EPOCHS):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network
            ###########################
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)

            # Real batch forward
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Fake batch forward
            noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network multiple times
            ###########################
            for _ in range(num_gen_steps):
                netG.zero_grad()
                # Use fresh noise each generator pass
                noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=device)
                fake = netG(noise)

                # We want to fool D, so label = real
                label.fill_(real_label)
                output = netD(fake).view(-1)
                errG = criterion(output, label)
                errG.backward()
                optimizerG.step()

            D_G_z2 = output.mean().item()

            # Print stats occasionally
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, EPOCHS, i, len(dataloader),
                       errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Record losses
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Generate fixed samples for visualization
            if (iters % 500 == 0) or ((epoch == EPOCHS - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    # Plot losses
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()


if __name__ == "__main__":
    main()
