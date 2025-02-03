import os
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.datasets as dset
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Subset, DataLoader

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
        super(Generator, self).__init__()
        
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

    # Load the full dataset
    full_dataset = torchvision.datasets.ImageFolder(
        root=DATA_PATH,
        transform=transform
    )
    
    # Create a subset with the first 200 images
    subset_indices = range(200)
    dataset = Subset(full_dataset, subset_indices)
    
    return dataset

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def learning_transfer(meta_model, model, transfer_ratio):
    """
    Transfer a fraction of parameters (based on sorted state_dict keys) from meta_model to model.
    The later layers (in the sorted order) are transferred.
    """
    meta_state = meta_model.state_dict()
    model_state = model.state_dict()
    keys = list(meta_state.keys())
    transfer_layer_idx = int(round(len(keys) * transfer_ratio))
    for i, key in enumerate(keys):
        if i > transfer_layer_idx:
            model_state[key].copy_(meta_state[key])
            # Optionally, freeze this parameter so it is not updated further.
            for name, param in model.named_parameters():
                if name == key:
                    param.requires_grad = False
    model.load_state_dict(model_state)
    return model

def meta_dataloader(dataset, batch_size, sample_ratio):
    dataset_len = len(dataset)
    sample_length = int(dataset_len * sample_ratio)
    random_list = random.sample(range(dataset_len), sample_length)
    meta_dataset = Subset(dataset, random_list)
    return DataLoader(meta_dataset, batch_size=batch_size, shuffle=True)

def meta_train_epoch(meta_epoch, meta_model, meta_train_dataloader, meta_loss_func, meta_optimizer, device, real_label, fake_label, LATENT_DIM, netG):
    meta_epoch_running_loss = 0
    meta_model.train()
    for meta_train_idx, data in enumerate(meta_train_dataloader, 0):
        features = data[0].to(device)
        b_size = features.size(0)

        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        
        meta_optimizer.zero_grad()

        # Use meta_model to process real images
        output = meta_model(features).view(-1)
        errD_real = meta_loss_func(output, label)
        errD_real.backward()

        noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = meta_model(fake.detach()).view(-1)
        errD_fake = meta_loss_func(output, label)
        errD_fake.backward()

        errD = errD_real + errD_fake
        meta_optimizer.step()
        
        meta_epoch_running_loss += errD.item()
        if meta_train_idx % 50 == 0:
            print(f'[Meta D] Meta Epoch {meta_epoch}, Batch {meta_train_idx}: Loss {errD.item()}')
    return meta_model, meta_epoch_running_loss / len(meta_train_dataloader)

def train_epoch(meta_epoch, epoch, model, train_dataloader, loss_func, optimizer, device, real_label, fake_label, LATENT_DIM, netG):
    epoch_running_loss = 0
    model.train()
    for train_idx, data in enumerate(train_dataloader, 0):
        features = data[0].to(device)
        b_size = features.size(0)

        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        
        optimizer.zero_grad()

        output = model(features).view(-1)
        errD_real = loss_func(output, label)
        errD_real.backward()

        noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = model(fake.detach()).view(-1)
        errD_fake = loss_func(output, label)
        errD_fake.backward()

        errD = errD_real + errD_fake
        optimizer.step()
        
        epoch_running_loss += errD.item()
        if train_idx % 50 == 0:
            print(f'[Train D] Meta Epoch {meta_epoch}, Epoch {epoch}, Batch {train_idx}: Loss {errD.item()}')
    return model, epoch_running_loss / len(train_dataloader)

def meta_train_epoch_gen(meta_epoch, meta_model, meta_train_dataloader, meta_loss_func, meta_optimizer, DEVICE, real_label, LATENT_DIM, netD):
    meta_epoch_running_loss = 0
    meta_model.train()
    for meta_train_idx, data in enumerate(meta_train_dataloader, 0):
        # Use batch size from the meta dataloader
        b_size = data[0].size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=DEVICE)
        meta_optimizer.zero_grad()
        noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=DEVICE)
        # Here we use the meta generator (not the main netG)
        fake = meta_model(noise)
        output = netD(fake).view(-1)
        errG = meta_loss_func(output, label)
        errG.backward()
        meta_optimizer.step()
    
        meta_epoch_running_loss += errG.item()
        if meta_train_idx % 50 == 0:
            print(f'[Meta G] Meta Epoch {meta_epoch}, Batch {meta_train_idx}: Loss {errG.item()}')
    return meta_model, meta_epoch_running_loss / len(meta_train_dataloader)

def train_epoch_gen(meta_epoch, epoch, model, train_dataloader, loss_func, optimizer, DEVICE, real_label, LATENT_DIM, netD):
    epoch_running_loss = 0
    model.train()
    for train_idx, data in enumerate(train_dataloader, 0):
        b_size = data[0].size(0)
        
        label = torch.full((b_size,), real_label, dtype=torch.float, device=DEVICE)
        optimizer.zero_grad()
        noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=DEVICE)
        fake = model(noise)
        output = netD(fake).view(-1)
        errG = loss_func(output, label)
        errG.backward()
        optimizer.step()
        
        epoch_running_loss += errG.item()
        if train_idx % 50 ==  0:
            print(f'[Train G] Meta Epoch {meta_epoch}, Epoch {epoch}, Batch {train_idx}: Loss {errG.item()}')
    return model, epoch_running_loss / len(train_dataloader)

def main():
    # Settings
    DATA_PATH = r"C:\Users\boezd\Documents\python\gan1\face_images"
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
    LATENT_DIM = 100
    meta_epochs = 2
    train_epochs = 1
    sample_ratio = 0.25
    transfer_ratio = 0.4

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("samples_gray", exist_ok=True)

    dataset = image_dataset(IMAGE_SIZE=IMAGE_SIZE, DATA_PATH=DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Initialize networks
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

    G_losses = []
    D_losses = []

    print("Starting Training Loop...")

    for epoch in range(EPOCHS):
        # --- Meta-Training for Discriminator ---
        meta_netD = copy.deepcopy(netD)
        meta_netD_optimizer = optim.Adam(meta_netD.parameters(), lr=LR, betas=(BETA1, BETA2))
        meta_netD_dataloader = meta_dataloader(dataset=dataset, batch_size=BATCH_SIZE, sample_ratio=sample_ratio)
        
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
        # Transfer a fraction of meta_netD parameters to netD
        netD = learning_transfer(meta_model=meta_netD, model=netD, transfer_ratio=transfer_ratio)
        
        for t_epoch in range(train_epochs):
            netD, train_loss = train_epoch(
                meta_epoch=epoch,
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

        # --- Meta-Training for Generator ---
        meta_netG = copy.deepcopy(netG)
        meta_netG_optimizer = optim.Adam(meta_netG.parameters(), lr=LR, betas=(BETA1, BETA2))
        meta_netG_dataloader = meta_dataloader(dataset=dataset, batch_size=BATCH_SIZE, sample_ratio=sample_ratio)
        
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
                netD=netD
            )
        # Transfer a fraction of meta_netG parameters to netG
        netG = learning_transfer(meta_model=meta_netG, model=netG, transfer_ratio=transfer_ratio)
        
        for t_epoch in range(train_epochs):
            netG, train_G_loss = train_epoch_gen(
                meta_epoch=epoch,
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

        G_losses.append(train_G_loss)
        D_losses.append(train_loss)

        # Save and visualize every few epochs
        if (epoch % 3 == 0):
            with torch.no_grad():
                netG.eval()
                fake = netG(fixed_noise).detach().cpu()
            
            img_grid = vutils.make_grid(fake, padding=2, normalize=True)
            save_image(img_grid, f"samples_gray/epoch_{epoch}.png")
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
