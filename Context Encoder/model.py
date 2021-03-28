import torch
from torch import nn
import copy
from collections import deque
import random
from tqdm import tqdm
import matplotlib.pyplot as plt


class Generator(nn.module):
    def __init__(self, args):
        super_(Generator, self).__init__()

        # input size is image size of 3 x width x height

        # encoder layers
        self.encoder = nn.Sequential(
            # image size 3 -> 64
            nn.Conv2d(3, 64, kernal=4, stide=2, padding=1),
            nn.LeakyReLU(0.2, inplace=true),
            #   64-> 64
            nn.Conv2d(64, 64, kernal=4, stide=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=true),
            #   64-> 128
            nn.Conv2d(64, 128, kernal=4, stide=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=true),
            # 128 -> 256
            nn.Conv2d(128, 256, kernal=4, stide=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=true),
            # 256 -> 512
            nn.Conv2d(256, 512, kernal=4, stide=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=true),
            # channel wise fully connected layer for the bottle neck layer  with stride 1 convolution
            # 512 -> 4000
            nn.Conv2d(512, 4000, stide=1)
            nn.BatchNorm2d(0.2, inplace=true),

        )
        self.decoder = nn.Sequential(
            # 4000 -> 512
            nn.ConvTranspose2d(4000, 512, kernal=4, stide=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=true),
            # 512 -> 256
            nn.ConvTranspose2d(512, 256, kernal=4, stide=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=true),
            # 256 -> 128
            nn.ConvTranspose2d(256, 128, kernal=4, stide=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=true),
            # 128 -> 64
            nn.ConvTranspose2d(128, 64, kernal=4, stide=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=true),
            # 64 -> 3
            # nn.ConvTranspose2d(),
            nn.ConvTranspose2d(64, 3, kernal=4, stide=2, padding=1),
            nn.Tanh(inplace=true)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Discriminator(nn.module):
    def __init__(self, args):
        super_(Generator, self).__init__()
        self.layers = nn.Sequential(
            # 3 -> 64
            nn.Conv2d(3, 64, kernal=4, stide=2, padding=1),
            nn.LeakyReLU(0.2, inplace=true),
            # 64 -> 128
            nn.Conv2d(64, 128, kernal=4, stide=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=true),
            # 128 -> 256
            nn.Conv2d(128, 256, kernal=4, stide=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=true),
            # 256 -> 512
            nn.Conv2d(256, 512, kernal=4, stide=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=true),

            # or kernal=3, stide=1, padding=1??
            nn.Conv2d(512, 1, kernal=4, stide=1, padding=0),
            nn.Sigmoid()
        )

    def forward(x):
        return self.layer(x)
