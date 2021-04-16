import cv2
import torch
import numpy as np


def get_area(img, thres=10000):
    """
    get text region of the image.
    https://stackoverflow.com/questions/37771263/detect-text-area-in-an-image-using-python-and-opencv
    """
    # Step 1: get binary image
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 30)

    # Step 2: combine adjacent text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
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
    mask[:, :, y: y + h, x: x + w] = 1.0
    return mask


def generate_mask_with_3_dimension(shape, area):
    """
    Generate mask in the given area
    """
    mask = torch.zeros(shape)
    x, y, w, h = area
    mask[y: y + h, x: x + w, :] = 1.0
    return mask


def extract_subtitle(img):
    """
    get text region of the image.
    https://www.programmersought.com/article/5117975415/
    """
    img = cv2.imread(img)
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    # only focus on the bottom
    x_end = int(2/3*img.shape[0])
    thresh[:x_end, :, :] = 0
    # print(thresh.shape)
    # thresh = np.where(thresh < 5, 0, 255)
    # thresh[thresh != 0] = 255
    # print((thresh < 5).all())
    kernel = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    cv2.imshow('thresh', thresh/255)

    cv2.imshow('dilate', dilate)

    cv2.waitKey()
    dilate[dilate != 0] = 1.0

    return dilate


def generate_mask_from_pos(shape, pos):
    """
    Generate mask in the given position
    """
    mask = torch.zeros(shape)
    # x_range, y_range, _ = pos.shape
    # for y in range(y_range):
    #     for x in range(x_range):
    #         if pos[x, y, 0] > 0:
    #             mask[:, :, x, y] = 1.0
    # print(torch.from_numpy(pos[:, :, 0]).shape)

    mask[:, :] = torch.from_numpy(pos[:, :, 0])
    return mask


def get_masked_area(img_path):
    x, y, w, h = get_area(img_path)
    image = cv2.imread(img_path)
    cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 3)
    # cv2.imshow('image', image)
    # cv2.waitKey()
    cv2.imwrite("./temp.jpg", image)

    img_path = "./temp.jpg"
    area = get_area(img_path)

    return area


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

    img_path = "GLCIC\with caption.PNG"
    x, y, w, h = get_area(img_path)
    image = cv2.imread(img_path)
    cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 3)
    # cv2.imshow('image', image)
    # cv2.waitKey()
    cv2.imwrite("GLCIC\with caption_masked.jpg", image)

    img_path = "GLCIC\with caption_masked.jpg"
    area = get_area(img_path)
    image = cv2.imread(img_path)
    mask = generate_mask_with_3_dimension(image.shape, area)
    result = np.where(mask == 1, 1, image)

    cv2.imwrite("GLCIC\with caption_masked_filled.jpg",
                result)
