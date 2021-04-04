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
import imageio
from dataprocess import DataProcess
import os
from PIL import Image
from skimage import img_as_ubyte
import numpy as np
import cv2

args_dict = {
    'image_size': 128,
    'lr': 0.00003,
    'beta1': 0.5,
    'beta2': 0.999,
    'batchSize': 32,
    'epoches': 50,
    'crop_size': 50
}

cuda = True


# Loss function
adversarial_loss = torch.nn.MSELoss()
adversarial_loss.cuda()
pixelwise_loss = torch.nn.L1Loss()
pixelwise_loss.cuda()


# Initialize generator and discriminator
Generator = Generator()
Discriminator = Discriminator()

Generator.cuda()
Discriminator.cuda()


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

dataset = dset.CIFAR10(root="ContextEncoder\Dataset", download=True,
                       transform=transforms.Compose([
                           transforms.Resize(args_dict["image_size"]),
                           transforms.ToTensor(),
                           transforms.Normalize(
                               (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                       ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args_dict["batchSize"],
                                         shuffle=True, num_workers=int(1))

# Optimizers
Gen_optimizer = torch.optim.Adam(Generator.parameters(
), lr=args_dict["lr"], betas=(args_dict["beta1"], args_dict["beta2"]))
Dis_optimizer = torch.optim.Adam(
    Discriminator.parameters(), lr=args_dict["lr"], betas=(args_dict["beta1"], args_dict["beta2"]))


def save_sample_image(Generator, real_image, iteration):

    SavingImage = real_image
    fake_center = Generator(SavingImage)
    SavingImage = SavingImage[0]

    # replace the center of the real_image with the fake_center
    fake_center_size = fake_center[0].shape[1]
    leftImageCenter = int(fake_center_size/2)
    SavingImage[:, leftImageCenter:leftImageCenter+fake_center_size,
                leftImageCenter: leftImageCenter+fake_center_size] = fake_center[0][:, :, :]
    path = os.path.join("ContextEncoder\Result",
                        'sample-{:06d}.png'.format(iteration))
    print(SavingImage.shape)

    result_image = torch.ones((128, 128, 3))
    result_image[:, :, 0] = SavingImage[0]
    result_image[:, :, 1] = SavingImage[1]
    result_image[:, :, 2] = SavingImage[2]
    result_image_1 = torch.ones((128, 128, 3), dtype=int)
    result_image_1 = result_image + 1
    result_image_1 *= 255/2
    imageio.imwrite(path, result_image_1.type(torch.uint8).detach())


if __name__ == '__main__':
    for epoch in range(args_dict["epoches"]):
        for i, image in enumerate(dataloader):
            dataclass = DataProcess(image, 64)
            masked_image, real_image = dataclass.process()
            real_image = real_image.cuda()
            masked_image = masked_image.cuda()
            # train discriminator
            Dis_optimizer.zero_grad()

            output = Discriminator(real_image.cuda()).cuda()
            real_label = torch.ones_like(output).cuda()
            real_loss = adversarial_loss(output, torch.ones_like(output))
            # D_x = torch.mean(real_loss)

            fake = Generator(real_image).cuda()
            fake_label = torch.zeros_like(fake).cuda()
            fake_loss = adversarial_loss(fake, fake_label)

            # detach or not?
            DGX = Discriminator(fake).cuda()
            # D_G_z1 = torch.mean(fake_loss)

            total_loss = real_loss + fake_loss

            total_loss.backward(retain_graph=True)
            Dis_optimizer.step()

            # maximize Generator

            # fake labels are real for generator loss accordin to paper implementation

            Gen_optimizer.zero_grad()
            real_label = torch.ones_like(DGX)
            G_adv_loss = adversarial_loss(DGX, real_label).cuda()

            G_pixel = pixelwise_loss(fake, masked_image).cuda()

            G_total_loss = 0.0001*G_adv_loss + 0.999*G_pixel

            G_total_loss.backward()
            Gen_optimizer.step()

            print("current epoch", epoch, "iteration", i, "total loss:", total_loss.item(),
                  "G total loss", G_total_loss.item())
            # save check point
            if(i % 100 == 0):
                save_sample_image(Generator, real_image, i)

        torch.save({'epoch': epoch+1,
                    'state_dict': Generator.state_dict()},
                   'model/netG_streetview.pth')
        torch.save({'epoch': epoch+1,
                    'state_dict': Discriminator.state_dict()},
                   'model/netlocalD.pth')
