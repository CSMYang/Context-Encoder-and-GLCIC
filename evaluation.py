import numpy
import math
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import torch


def psnr(real_img, generated_img):
    """
    This function computes the PSNR score between two images.
    :param generated_img: A tensor to the generated image
    :param real_img: A tensor to the real image
    :return: The PSNR score
    """
    img_1 = real_img.numpy()
    img_2 = generated_img.numpy()

    return peak_signal_noise_ratio(img_1, img_2, data_range=real_img.max() - real_img.min())


def ssim(real_img, generated_img):
    """
    The function computes the SSIM score between two images.
    :param real_img: A tensor to the generated image
    :param generated_img: A tensor to the real image
    :return: The structural similarity score
    """
    img_1 = real_img.numpy()
    img_2 = generated_img.numpy()

    return structural_similarity(img_1, img_2, data_range=real_img.max() - real_img.min())