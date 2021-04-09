"""
Do comparison between ContextEncoder and GLCIC
"""
import cv2
import os
import numpy as np
import scipy
# from fid_score.fid_score import FidScore
from keras.applications.inception_v3 import InceptionV3, preprocess_input

# def calculate_fid(source_img_dir, target_img_dir, num):
#     """
#     Calculate FID score.
#     """
#     paths = [source_img_dir, target_img_dir]
#     fid = FidScore(paths, torch.device('cuda:0'), num)
#     score = fid.calculate_fid_score()
#     return score


def get_images(image_dir, shape):
    """
    Return a ndarray containing all images from image_dir.
    """
    images = []
    for image in os.listdir(image_dir):
        img = cv2.resize(cv2.imread(os.path.join(image_dir, image)), shape)
        if img is not None:
            images.append(img)
    return np.asarray(images).astype('float32')


def calculate_fid(source_img_dir, target_img_dir, shape=(299, 299, 3)):
    """
    Calculate FID score.
    https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
    """
    model = InceptionV3(include_top=False, pooling='avg', input_shape=shape)
    # read the images
    size = (shape[0], shape[1])
    imgs1, imgs2 = get_images(source_img_dir, size), get_images(target_img_dir, size)
    act1, act2 = model.predict(preprocess_input(imgs1)), model.predict(preprocess_input(imgs2))
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    score = np.sum((mu1 - mu2)**2.0) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return score


if __name__ == '__main__':
    # test
    img_real_dir = ""
    output_dir = ""
    fid = calculate_fid(img_real_dir, output_dir)
    print('FID (same): %.3f' % fid)
    # fid between images1 and images2
    fid = calculate_fid(img_real_dir, img_real_dir)
    print('FID (different): %.3f' % fid)
