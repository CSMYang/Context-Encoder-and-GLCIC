import cv2
import torch
import numpy as np


def get_area(img, thres=10000):
    """
    get text region of the image.
    https://stackoverflow.com/questions/37771263/detect-text-area-in-an-image-using-python-and-opencv
    """
    # Step 1: get binary image
    image = img
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 30)

    # Step 2: combine adjacent text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Step 3: find appropriate subtitle area
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # for c in cnts:
    #     area = cv2.contourArea(c)
    #     if area > thres:
    #         x, y, w, h = cv2.boundingRect(c)
    #         # cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 3)
    best_c = None
    best_area = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if best_area is None or area > best_area:
            best_area = area
            best_c = c

    # x, y, w, h = cv2.boundingRect(best_c)
    # cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 3)
    # cv2.imshow('thresh', thresh)
    # cv2.imshow('dilate', dilate)
    # cv2.imshow('image', image)
    # cv2.waitKey()

    return cv2.boundingRect(best_c)


def generate_mask(shape, area):
    """
    Generate mask in the given area
    """
    mask = torch.zeros(shape)
    x, y, w, h = area
    mask[:, y: y + h, x: x + w, :] = 1.0
    return mask


if __name__ == "__main__":

    # test get_area
    # for i in range(1000):
    #     img_path = "C:/Users/apple1/Downloads/CSC413H1S/Project/my work/data/{}.png".format(i)
    #     x, y, w, h = get_area(img_path)
    #     image = cv2.imread(img_path)
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 3)
    #     # cv2.imshow('image', image)
    #     # cv2.waitKey()
    #     cv2.imwrite("C:/Users/apple1/Downloads/CSC413H1S/Project/my work/result/{}.png".format(i), image)

    # test generate mask
    # for i in range(1):
    # img_path = "C:/Users/apple1/Downloads/CSC413H1S/Project/my work/data/{}.png".format(
    #     i)
    # area = get_area(img_path)
    # image = cv2.imread(img_path)
    # result = generate_mask(image.shape, area)
    # cv2.imshow('image', result)
    # cv2.waitKey()

    img_path = "GLCIC\movie_caption.jpg"
    x, y, w, h = get_area(img_path)
    image = cv2.imread(img_path)
    cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 3)
    # cv2.imshow('image', image)
    # cv2.waitKey()
    cv2.imwrite("GLCIC\caption_masked.jpg", image)

    img_path = "GLCIC\caption_masked.jpg"
    area = get_area(img_path)
    image = cv2.imread(img_path)
    mask = generate_mask(image.shape, area)
    result = np.where(mask, 1, image)

    cv2.imwrite("GLCIC\caption_masked_filled.jpg",
                result)
