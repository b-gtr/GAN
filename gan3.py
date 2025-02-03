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

from torch.utils.data import Subset, Dataset, DataLoader

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

def image_dataset(IMAGE_SIZE, DATA_PATH):
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
    return dataset

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def learning_transfer(meta_model, model, transfer_ratio):
    i = 0
    for _ in meta_model.named_parameters():
        i += 1
    transfer_layer_idx = int(round(i * transfer_ratio))

    i = 0
    for param_first, param_second in zip(meta_model.parameters(), model.parameters()):
        if i > transfer_layer_idx:
            param_second.data.copy_(param_first.data)
            param_second.requires_grad = False
        i += 1
    return model

def grad_to_true(model):
    for param in model.parameters():
        param.requires_grad = True
    return model

def meta_dataloader(dataset, batch_size, sample_ratio):
    dataset_len = len(dataset)
    sample_length = int(dataset_len * sample_ratio)
    random_list = random.sample(range(dataset_len), sample_length)  # Fixed range starting from 0
    meta_dataset = Subset(dataset, random_list)
    return DataLoader(meta_dataset, batch_size=batch_size, shuffle=True)

def meta_train_epoch(meta_epoch, meta_model, meta_train_dataloader, meta_loss_func, meta_optimizer, device, real_label, fake_label, LATENT_DIM, netG):
    meta_epoch_running_loss = 0
    for meta_train_idx, data in enumerate(meta_train_dataloader, 0):
        features = data[0].to(device)
        b_size = features.size(0)

        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        
        meta_optimizer.zero_grad()

        output = meta_model(features).view(-1)
        errD_real = meta_loss_func(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = meta_model(fake.detach()).view(-1)
        errD_fake = meta_loss_func(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        meta_optimizer.step()
        
        meta_epoch_running_loss += errD.item()
        if meta_train_idx % 50 == 0:
            print(f'Meta Epoch {meta_epoch}, Batch {meta_train_idx}: Loss {errD.item()}')
    return meta_model, meta_epoch_running_loss / len(meta_train_dataloader)

def train_epoch(meta_epoch, epoch, model, train_dataloader, loss_func, optimizer, device, real_label, fake_label, LATENT_DIM, netG):
    epoch_running_loss = 0
    for train_idx, data in enumerate(train_dataloader, 0):
        features = data[0].to(device)
        b_size = features.size(0)

        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        
        optimizer.zero_grad()

        output = model(features).view(-1)
        errD_real = loss_func(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = model(fake.detach()).view(-1)
        errD_fake = loss_func(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake
        optimizer.step()
        
        epoch_running_loss += errD.item()
        if train_idx % 50 == 0:
            print(f'Meta Epoch {meta_epoch}, Epoch {epoch}, Batch {train_idx}: Loss {errD.item()}')
    return model, epoch_running_loss / len(train_dataloader)

def meta_train_epoch_gen(meta_epoch,meta_model, meta_train_dataloader, meta_loss_func, meta_optimizer, DEVICE, real_label, LATENT_DIM, netG, netD):
    meta_epoch_running_loss = 0
    for meta_train_idx, data in enumerate(meta_train_dataloader, 0):
        features = data[0].to(DEVICE)
        b_size = features.size(0)

        label = torch.full((b_size,), real_label, dtype=torch.float, device=DEVICE)
        meta_optimizer.zero_grad()
        label.fill_(real_label)
        noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=DEVICE)
        fake = netG(noise)
        output = netD(fake).view(-1)
        errG = meta_loss_func(output, label)
        errG.backward()
        meta_optimizer.step()
    
        meta_epoch_running_loss += errG.item()
        if meta_train_idx % 50 == 0:
            print(f'the meta training batch loss at meta epoch of  {meta_epoch} idx of {meta_train_idx} is {errG}')
    return meta_model, meta_epoch_running_loss / len(meta_train_dataloader)



def train_epoch_gen(meta_epoch, epoch, model, train_dataloader, loss_func, optimizer, DEVICE, real_label, LATENT_DIM, netD):
    epoch_running_loss = 0
    for train_idx, data in enumerate(train_dataloader, 0):
        features = data[0].to(DEVICE)
        b_size = features.size(0)
        
        label = torch.full((b_size,), real_label, dtype=torch.float, device=DEVICE)
        optimizer.zero_grad()
        label.fill_(real_label)
        noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=DEVICE)
        fake = model(noise)
        output = netD(fake).view(-1)
        errG = loss_func(output, label)
        errG.backward()
        optimizer.step()
        
        
        epoch_running_loss += errG.item()
        if train_idx % 50 ==  0:
            print(f'the training batch loss at meta epoch {meta_epoch} epoch {epoch} idx of {train_idx} is {errG}')
    return model, epoch_running_loss / len(train_dataloader)


def main():
    DATA_PATH = r"pfad"
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
    meta_epochs = 2
    train_epochs = 1
    sample_ratio = 0.25
    transfer_ratio = 0.4

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("samples_gray", exist_ok=True)

    dataset = image_dataset(IMAGE_SIZE=IMAGE_SIZE, DATA_PATH=DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    netG = Generator(LATENT_DIM, NGF, NC).to(DEVICE)
    netD = Discriminator(NC, NDF).to(DEVICE)

    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, BETA2))
    optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, BETA2))

    fixed_noise = torch.randn(BATCH_SIZE, LATENT_DIM, 1, 1, device=DEVICE)

    real_label = 1.0
    fake_label = 0.0

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")

    num_gen_steps = 3  # Update generator multiple times per discriminator step

    for epoch in range(EPOCHS):
        netD.zero_grad()
        meta_netD = grad_to_true(netD)
        meta_netD_optimizer = optim.Adam(meta_netD.parameters(), lr=LR, betas=(BETA1, BETA2))
        meta_netD_dataloader = meta_dataloader(dataset=dataset, batch_size=BATCH_SIZE, sample_ratio=sample_ratio)
        meta_netD.train()

        for meta_epoch in range(meta_epochs): 
            meta_netD, meta_loss = meta_train_epoch(
                meta_epoch=meta_epoch,
                meta_model=meta_netD,
                meta_train_dataloader=meta_netD_dataloader,
                meta_loss_func=criterion,
                meta_optimizer=meta_netD_optimizer,
                device=DEVICE,
                real_label=real_label,
                fake_label=fake_label,
                LATENT_DIM=LATENT_DIM,
                netG=netG
            )

        netD = learning_transfer(meta_model=meta_netD, model=netD, transfer_ratio=transfer_ratio)
        netD.train()
        
        for t_epoch in range(train_epochs):
            netD, train_loss = train_epoch(
                meta_epoch=meta_epoch,
                epoch=t_epoch,
                model=netD,
                train_dataloader=dataloader,
                loss_func=criterion,
                optimizer=optimizerD,
                device=DEVICE,
                real_label=real_label,
                fake_label=fake_label,
                LATENT_DIM=LATENT_DIM,
                netG=netG
            )

        
        #Generator part
        netG.zero_grad()
        meta_netG = grad_to_true(netD)
        meta_netG_optimizer = optim.Adam(meta_netG.parameters(), lr=LR, betas=(BETA1, BETA2))
        meta_netG_dataloader = meta_dataloader(dataset=dataset, batch_size=BATCH_SIZE, sample_ratio=sample_ratio)
        meta_netG.train()

        for meta_epoch in range(meta_epochs): 
            meta_netG, meta_G_loss = meta_train_epoch_gen(
                meta_epoch=meta_epoch,
                meta_model=meta_netG,
                meta_train_dataloader=meta_netG_dataloader,
                meta_loss_func=criterion,
                meta_optimizer=meta_netG_optimizer,
                DEVICE=DEVICE,
                real_label=real_label,
                LATENT_DIM=LATENT_DIM,
                netG=netG,
                netD=netD
            )

        netG = learning_transfer(meta_model=meta_netG, model=netG, transfer_ratio=transfer_ratio)
        netG.train()

        for t_epoch in range(train_epochs):
            netG, train_G_loss = train_epoch_gen(
                meta_epoch=meta_epoch,
                epoch=t_epoch,
                model=netG,
                train_dataloader=dataloader,
                loss_func=criterion,
                optimizer=optimizerG,
                DEVICE=DEVICE,
                real_label=real_label,
                LATENT_DIM=LATENT_DIM,
                netD=netD
            )

        G_losses.append(train_G_loss.item())
        D_losses.append(train_loss.item())

        if (epoch % 3 == 0):
            with torch.no_grad():
                netG.eval()
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            netG.train()



    # Plot training losses
    plt.figure(figsize=(10,5))
    plt.title("Training Losses")
    plt.plot(G_losses, label="Generator")
    plt.plot(D_losses, label="Discriminator")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
