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


# takes in a batch of image as input
class DataProcess():
    def __init__(self, image, mask_size):

        self.image_batch = image
        self.mask_size = mask_size
        self.image_size = image[0].shape[1]  # the width/height of image

    def process(self):
        images, _ = self.image_batch
        images.cuda()
        leftImageCenter = int(self.mask_size/2)
        masked_image = images[:, :, leftImageCenter:leftImageCenter +
                              self.mask_size, leftImageCenter:leftImageCenter + self.mask_size].cuda()

        return masked_image, images
