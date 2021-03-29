import torch
from torch import nn
import copy
from collections import deque
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.datasets as dset
from model import Generator, Discriminator
import torchvision.transforms as transforms


args_dict = {
    'image_size': 128,
    'lr': 0.00003,
    'beta1': 0.5,
    'beta2': 0.999,
    'batch_size': 32,
    'epoches': 50,
    'crop_size': 50
}

cuda = True if torch.cuda.is_available() else False


# Loss function
adversarial_loss = torch.nn.BCELoss()
adversarial_loss.cuda()
pixelwise_loss = torch.nn.L1Loss()
pixelwise_loss.cuda()


# Initialize generator and discriminator
Generator = Generator()
Discriminator = Discriminator()


# # Dataset loader
# transforms_ = [
#     transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ]
# dataloader = DataLoader(
#     ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_),
#     batch_size=opt.batch_size,
#     shuffle=True,
#     num_workers=opt.n_cpu,
# )
# test_dataloader = DataLoader(
#     ImageDataset("../../data/%s" % opt.dataset_name,
#                  transforms_=transforms_, mode="val"),
#     batch_size=12,
#     shuffle=True,
#     num_workers=1,
# )
dataset = dset.CIFAR10(root="Context Encoder\Dataset", classes=['bedroom_train'],
                       transform=transforms.Compose([
                           transforms.Resize(args_dict["image_size"]),
                           transforms.CenterCrop(args_dict["crop_size"]),
                           transforms.ToTensor(),
                           transforms.Normalize(
                            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                       ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args_dict["batchSize"],
                                         shuffle=True, num_workers=int(opt.workers))

# Optimizers
Gen_optimizer = torch.optim.Adam(generator.parameters(
), lr=args_dict["lr"], betas=(args_dict["beta1"], args_dict["beta2"]))
Dis_optimizer = torch.optim.Adam(
    discriminator.parameters(), lr=args_dict["lr"], betas=(args_dict["beta1"], args_dict["beta2"]))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


for epoch in args_dict["epoches"]:
    for i, image in enumerate(dataloader):
        masked_image = image

        # train discriminator
        Dis_optimizer.zero_grad()

        output = Discriminator(real_center)
        real_label = torch.ones_like(output)
        real_loss = adversarial_loss(output, torch.ones_like(output))
        D_x = torch.mean(real_loss)

        fake = Generator(masked_image)
        fake_label = torch.zeros_like(fake)
        # detach or not?
        DGX = Discriminator(fake.detach())

        fake_loss = adversarial_loss(output, fake_label)

        D_G_z1 = torch.mean(fake_loss)

        total_loss = real_loss + fake_loss

        total_loss.backward()

        Dis_optimizer.step()

        # maximize Generator

        # fake labels are real for generator loss accordin to paper implementation

        Gen_optimizer.zero_grad()

        G_adv_loss = adversarial_loss(DGX, real_label)

        G_pixel = pixelwise_loss(fake_images, masked_parts)

        G_total_loss = 0.0001*G_adv_loss + 0.999*G_pixel

        G_total_loss.backward()

        Gen_optimizer.step()
