import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Discriminator(nn.Module):
    def __init__(self, input_channel, ndf):
        super(Discriminator,self).__init__()

        self.discriminator = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(input_channel, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
    
    @staticmethod
    def linear_block(in_ftrs,out_ftrs,p):
        return nn.Sequential(
            nn.Linear(in_ftrs,out_ftrs),
            nn.BatchNorm1d(out_ftrs),
            nn.ReLU(),
            nn.Dropout(p)
        )

        
    def forward(self,inp):
        return self.discriminator(inp)


class Generator(nn.Module):
    def __init__(self, latent_space, ngf, input_channel):
        
        super(Generator,self).__init__()
        
        self.generator = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d( latent_space, 1024, 4, 1, 0, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),

                # state size. (ngf) x 64 x 64
                nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),


                # state size. (ngf) x 128 x 128
                nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(True),

                # state size. (ngf) x 256 x 256
                nn.ConvTranspose2d(16, input_channel, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. input_channel x 512 x 512
            )

    def forward(self,inp):
        return self.generator(inp)
