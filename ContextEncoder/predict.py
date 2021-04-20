import copy
import imghdr
import os
import random
from collections import deque

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image
from skimage import img_as_ubyte
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from dataprocess import DataProcess
from model import Discriminator, Generator
import numpy
import math
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import torch


def psnr(real_img, generated_img, PIXEL_MAX=255.0):
    """
    This function computes the PSNR score between two images.
    :param generated_img: A tensor to the generated image
    :param real_img: A tensor to the real image
    :return: The PSNR score
    """
    img_1 = real_img.cpu().numpy()
    img_2 = generated_img.cpu().numpy()
    mse = numpy.mean((img_1 - img_2) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def ssim(real_img, generated_img):
    """
    The function computes the SSIM score between two images.
    :param real_img: A tensor to the generated image
    :param generated_img: A tensor to the real image
    :return: The structural similarity score
    """
    img_1 = real_img.cpu().numpy()
    img_2 = generated_img.cpu().numpy()

    return structural_similarity(img_1, img_2, multichannel=True)


def visualize_saliency_map(img_path, input_width, input_height, model):
    """
    This function shows the saliency map of an image
    :param img_path: a string of the path to the image for visualization
    :param input_width: input height to the model
    :param input_height: input width to the model
    :param model: the model used for image classification. In our case, it will be a discriminator
    """
    image = Image.open(img_path)
    transform = transforms.Compose([
        transforms.CenterCrop(input_width),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])

    image = transform(image)
    image.reshape(1, 3, input_height, input_width)
    image = image.cuda()
    image.requires_grad_()
    temp_img = torch.zeros((1, 3, 64, 64))
    temp_img[0] = image
    temp_img.cuda()
    temp_img.requires_grad_()
    temp_img.retain_grad()
    output = model(temp_img)
    output_idx = output.argmax()
    output_max = output[0, output_idx]
    output_max.backward()
    saliency, _ = torch.max(temp_img.grad.data.abs(), dim=1)
    saliency = saliency.reshape(input_height, input_width)

    temp_img = temp_img[0].reshape(-1, input_height, input_width)
    temp_image_1 = torch.ones((3, 64, 64), dtype=int)
    temp_image_1 = temp_img + 1
    temp_image_1 *= 255/2
    print(temp_image_1.cpu().detach().numpy().transpose(1, 2, 0))
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(temp_image_1.cpu().detach(
    ).numpy().astype(np.int).transpose(1, 2, 0))
    ax[0].axis('off')
    ax[1].imshow(saliency.cpu(), cmap='hot')
    ax[1].axis('off')
    plt.tight_layout()
    plt.show()


def predict(Generator, real_image):

    SavingImage = real_image.clone()
    fake_center = Generator(SavingImage)
    SavingImage = SavingImage[0]

    # replace the center of the real_image with the fake_center
    fake_center_size = fake_center[0].shape[1]
    leftImageCenter = int(fake_center_size/2)
    SavingImage[:, leftImageCenter:leftImageCenter+fake_center_size,
                leftImageCenter: leftImageCenter+fake_center_size] = fake_center[0][:, :, :]
    result_image = torch.ones((128, 128, 3))
    result_image[:, :, 0] = SavingImage[0]
    result_image[:, :, 1] = SavingImage[1]
    result_image[:, :, 2] = SavingImage[2]
    result_image_1 = torch.ones((128, 128, 3), dtype=int)
    result_image_1 = result_image + 1
    result_image_1 *= 255/2

    imageio.imwrite("ContextEncoder\\testsample.png",
                    result_image_1.type(torch.uint8).detach())

    temp_image = torch.ones((128, 128, 3))
    temp_image[:, :, 0] = real_image[0][0]
    temp_image[:, :, 1] = real_image[0][1]
    temp_image[:, :, 2] = real_image[0][2]
    temp_image_1 = torch.ones((128, 128, 3), dtype=int)
    temp_image_1 = temp_image + 1
    temp_image_1 *= 255/2

    imageio.imwrite("ContextEncoder\\testresult.png",
                    temp_image_1.type(torch.uint8).detach())
    return result_image_1


"""
https://github.com/otenim/GLCIC-PyTorch/blob/master/datasets.py
"""


class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None, recursive_search=False):
        super(ImageDataset, self).__init__()
        self.data_dir = os.path.expanduser(data_dir)
        self.transform = transform
        self.images = self.__load_images_from_dir(
            self.data_dir, walk=recursive_search)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index, color_format='RGB'):
        img = (Image.open(self.images[index])).convert(color_format)
        return self.transform(img) if self.transform else img

    def __is_imgfile(self, filepath):
        filepath = os.path.expanduser(filepath)
        return os.path.isfile(filepath) and imghdr.what(filepath)

    def __load_images_from_dir(self, dirpath, walk=False):
        images = []
        dirpath = os.path.expanduser(dirpath)
        if walk:
            for (root, _, files) in os.walk(dirpath):
                for file in files:
                    file = os.path.join(root, file)
                    if self.__is_imgfile(file):
                        images.append(file)
        else:
            for path in os.listdir(dirpath):
                path = os.path.join(dirpath, path)
                if not self.__is_imgfile(path):
                    continue
                images.append(path)
        return images


if __name__ == '__main__':
    # load discriminator model
    Discriminator = Discriminator()
    Discriminator.load_state_dict(
        torch.load("ContextEncoder\model\Discriminator\Discriminator_CelebA_2.pth"))
    # visualize_saliency_map(
    #     "ContextEncoder\TestResultForFid\Fake\\1_generated.png", 64, 64, Discriminator)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])
    Generator = Generator().cuda()
    Generator.load_state_dict(
        torch.load("ContextEncoder\model\Generator\Generator_CelebA_2.pth")
    )
    # read image in folder
    train_set = ImageDataset("img_align_celeba\\test",
                             transform, recursive_search=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                               shuffle=False)

    # train_set = ImageDataset("ContextEncoder\\testing_image\\real",
    #                          transform, recursive_search=True)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
    #                                            shuffle=False)

    # read CIFAR10 dataset
    # dataset = dset.CIFAR10(root="ContextEncoder\Dataset", download=True,
    #                        transform=transforms.Compose([
    #                            transforms.Resize(128),
    #                            transforms.ToTensor(),
    #                            transforms.Normalize(
    #                                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                        ]))

    # train_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
    #                                            shuffle=False, num_workers=int(1))

    total_ssim = 0
    total_psnr = 0
    for i, img in enumerate(train_loader):

        if(i == 10):
            break
        image = img

        real_image = image.clone()
        image[:, :, 32:32+64,
              32:32+64] = 0

        image = image.cuda()

        temp_img = torch.zeros((1, 3, 128, 128))
        temp_img = image.clone()
        temp_img.cuda()

        result = predict(Generator, temp_img)

        # real masked image
        temp_temp_image = torch.zeros((128, 128, 3))
        temp_temp_image[:, :, 0] = temp_img[0][0]
        temp_temp_image[:, :, 1] = temp_img[0][1]
        temp_temp_image[:, :, 2] = temp_img[0][2]
        temp_temp_image_1 = torch.zeros((128, 128, 3), dtype=int)
        temp_temp_image_1 = temp_temp_image + 1
        temp_temp_image_1 *= 255/2

        # real image
        real_real_image = torch.zeros((128, 128, 3))
        real_real_image[:, :, 0] = real_image[0][0]
        real_real_image[:, :, 1] = real_image[0][1]
        real_real_image[:, :, 2] = real_image[0][2]
        real_real_image_1 = torch.zeros((128, 128, 3), dtype=int)
        real_real_image_1 = real_real_image + 1
        real_real_image_1 *= 255/2

        imageio.imwrite("ContextEncoder\\TestResultForFid\\Real\\{}_real.png".format(i),
                        real_real_image_1.type(torch.uint8).detach())

        imageio.imwrite("ContextEncoder\\TestResultForFid\\Fake\\{}_masked_real.png".format(i),
                        temp_temp_image_1.type(torch.uint8).detach())

        imageio.imwrite("ContextEncoder\\TestResultForFid\\Fake\\{}_generated.png".format(i),
                        result.type(torch.uint8).detach())

        # print(ssim(result.cpu().detach(), temp_temp_image_1))
        ssim_result = ssim(result[32:32+64,
                                  32:32+64, :].cpu().detach(), real_real_image_1[32:32+64,
                                                                                 32:32+64, :])

        psnr_result = psnr(result[32:32+64,
                                  32:32+64, :].cpu().detach(), real_real_image_1[32:32+64,
                                                                                 32:32+64, :])
        total_psnr += psnr_result
        total_ssim += ssim_result
        print("".format(i), "ssim :", ssim_result, "psnr:", psnr_result)

print("ssim:", total_ssim/10, "psnr", total_psnr/10)
