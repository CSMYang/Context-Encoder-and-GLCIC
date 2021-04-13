import json
import os
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from model_pretrained import CompletionNetwork, ContextDiscriminator
from train import poisson_blend, crop, generate_area
from dataset import ImageDataset
from detect_subtitle import get_area, generate_mask, extract_subtitle, generate_mask_from_pos
from vanilla_gradient import visualize_saliency_map
from matplotlib import pyplot as plt
import numpy as np
import glob
import cv2


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def make_video(args, mpv, model):
    """
    Extract subtitles for all images from dir and make the video.
    """
    # read images
    img_array = []
    for filename in glob.glob('{}/*.{}'.format(args.input_img, args.img_type)):
        img_array.append(filename)
    # mask_path = args.output_img + "/mask"
    # output_path = args.output_img + "/output"
    size = (img_array[0].shape[0], img_array[0].shape[1])
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('{}/test.avi'.format(args.output_img), fourcc, 3, size)

    for img_path in range(img_array):
        img = Image.open(img_path)
        # img = transforms.Resize(args.img_size)(img)
        # img = transforms.RandomCrop((args.img_size, args.img_size))(img)
        x = transforms.ToTensor()(img)
        x = torch.unsqueeze(x, dim=0)[:, 0:3, :, :]

        # create mask
        temp_path = "./test.png"
        save_image(x, temp_path, nrow=1)
        if args.method:
            mask = generate_mask(
                shape=(1, 1, x.shape[2], x.shape[3]),
                area=get_area(temp_path)
            )
        else:
            mask = generate_mask_from_pos(
                shape=(1, 1, x.shape[2], x.shape[3]),
                pos=extract_subtitle(temp_path)
            )

        # inpaint
        model.eval()
        with torch.no_grad():
            x_mask = x - x * mask + mpv * mask
            input = torch.cat((x_mask, mask), dim=1)
            output = model(input)
            inpainted = poisson_blend(x_mask, output, mask)
            # imgs = torch.cat((x, x_mask, inpainted), dim=0)
            # imgs = inpainted
        #     save_image(imgs, args.output_img, nrow=3)
        # print('output img was saved as %s.' % args.output_img)
            out.write(inpainted)
            os.remove(temp_path)
    out.release()


if __name__ == "__main__":
    args = AttrDict()
    # set hyperparameters
    args_dict = {
        "model": "GLCIC\pretrained_model_cn",
        "config": "GLCIC\config.json",
        "input_img": "GLCIC\movie_caption.jpg",  # input img
        "img_type": "png",
        "output_img": "GLCIC\\result.jpg",  # output img name
        "method": True, # True for generate mask, False for generate mask from pos
        "max_holes": 5,
        "img_size": 160,
        "hole_min_w": 24,
        "hole_max_w": 46,
        "hole_min_h": 24,
        "hole_max_h": 48,
    }
    args.update(args_dict)

    # =============================================
    # Load model
    # =============================================
    with open(args.config, 'r') as f:
        config = json.load(f)
    mpv = torch.tensor(config['mpv']).view(1, 3, 1, 1)
    model = CompletionNetwork()
    model.load_state_dict(torch.load(args.model, map_location='cpu'))

    # =============================================
    # Predict
    # =============================================
    # convert img to tensor
    img = Image.open(args.input_img)
    img = transforms.Resize(args.img_size)(img)
    # img = transforms.RandomCrop((args.img_size, args.img_size))(img)
    x = transforms.ToTensor()(img)
    x = torch.unsqueeze(x, dim=0)[:, 0:3, :, :]

    # create mask
    temp_path = "./test.png"
    save_image(x, temp_path, nrow=1)
    mask = generate_mask(
        shape=(1, 1, x.shape[2], x.shape[3]),
        area=get_area(temp_path)
    )

    # inpaint
    model.eval()
    with torch.no_grad():
        x_mask = x - x * mask + mpv * mask
        input = torch.cat((x_mask, mask), dim=1)
        output = model(input)
        inpainted = poisson_blend(x_mask, output, mask)
        # imgs = torch.cat((x, x_mask, inpainted), dim=0)
        imgs = inpainted
        save_image(imgs, args.output_img, nrow=3)
    print('output img was saved as %s.' % args.output_img)

    x, y, w, h = get_area(temp_path)
    while(y+96 > 160):
        y -= 1
    while(x+96 > 160):
        x -= 1
    area = ((x, y), (96, 96))
    hole_area_fake = generate_area((96, 96),
                                   (inpainted.shape[3], inpainted.shape[2]))

    input_ld_fake = crop(inpainted, area)
    CD = ContextDiscriminator(local_input_shape=input_ld_fake[0].shape,
                              global_input_shape=(
                                  3, 160, 160),
                              arc='celeba')
    CD.load_state_dict(
        state_dict=torch.load("GLCIC\pretrained_model_cd"))

    CD = CD.cuda()
    plt.imshow(input_ld_fake[0].numpy().astype(np.float32).transpose(1, 2, 0))
    plt.show()
    visualize_saliency_map("GLCIC\\result.jpg", input_ld_fake, 160, 160, CD)
    os.remove(temp_path)
