import numpy
import math
from skimage.metrics import structural_similarity


def psnr(real_img, generated_img, PIXEL_MAX = 255.0):
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

    return structural_similarity(img_1, img_2, data_range=real_img.max() - real_img.min())