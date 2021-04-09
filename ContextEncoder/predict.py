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


if __name__ == '__main__':
    # load discriminator model
    Discriminator = Discriminator()
    Discriminator.load_state_dict(
        torch.load("ContextEncoder\model\Discriminator\Discriminator_9.pth"))
    visualize_saliency_map(
        "ContextEncoder\Result_1\sample-001500 -000007.png", 64, 64, Discriminator)
