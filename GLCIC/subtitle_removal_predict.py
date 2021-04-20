import json
import os
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from model_pretrained import CompletionNetwork, ContextDiscriminator
from train import poisson_blend, crop, generate_area
from dataset import ImageDataset
from detect_subtitle import get_area, generate_mask, get_masked_area, extract_subtitle, generate_mask_from_pos
from vanilla_gradient import visualize_saliency_map
from matplotlib import pyplot as plt
import numpy as np
from evaluation import ssim, psnr
import glob
import cv2


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def make_video(args, mpv, model, img_type="png"):
    """
    Predict imgs from dir
    """
    # convert img to tensor
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(
        '{}/test.avi'.format(args.output_dir), fourcc, args.fps, args.video_size)
    onlyfiles = next(os.walk(args.input_dir))[2]
    pbar = tqdm(total=len(onlyfiles),
                desc='Removing subtitles...')
    for filename in glob.glob('{}/*.{}'.format(args.input_dir, img_type)):
        img = Image.open(filename)
        # img = transforms.Resize((args.img_size))(img)
        x = transforms.ToTensor()(img)
        x = torch.unsqueeze(x, dim=0)[:, 0:3, :, :]

        # create mask
        temp_path = "./test.jpg"
        save_image(x, temp_path, nrow=1)
        if args.method:
            area = get_masked_area(temp_path)
            mask = generate_mask(
                shape=(1, 1, x.shape[2], x.shape[3]),
                area=area
            )
        else:
            area = extract_subtitle(temp_path)
            mask = generate_mask_from_pos(
                shape=(1, 1, x.shape[2], x.shape[3]),
                pos=area
            )
        # inpaint
        os.remove(temp_path)
        model.eval()
        with torch.no_grad():
            x_mask = x - x * mask + mpv * mask
            input = torch.cat((x_mask, mask), dim=1)
            output = model(input)
            frame = poisson_blend(x_mask, output, mask)
            # maskpath = os.path.join(mask_path, 'test_%d.png' % i)
            # outputpath = os.path.join(output_path, 'test_%d.png' % i)
            # save_image(imgs, maskpath)
            save_image(frame, temp_path)
            # print("image added+1")
            out.write(cv2.imread(temp_path))
            pbar.update()
            os.remove(temp_path)
    out.release()
    pbar.close()
    print('output video was saved as %s.' % args.output_dir)


def image_convert_shape(image):
    temp_image = torch.zeros((image.shape[1], image.shape[2], image.shape[0]))
    temp_image[:, :, 0] = image[0]
    temp_image[:, :, 1] = image[1]
    temp_image[:, :, 2] = image[2]

    return temp_image


if __name__ == "__main__":
    args = AttrDict()
    # set hyperparameters
    args_dict = {
        "model": "GLCIC\pretrained_model_cn",
        "config": "GLCIC\config.json",
        "input_img": "GLCIC\\netflix_2_with_caption.png",  # input img
        "output_img": "GLCIC\\result.jpg",  # output img name
        "input_img2": "GLCIC\\netflix_2_with_caption.png",
        "input_dir": "GLCIC\\video_frames",  # input img directory
        "output_dir": "GLCIC\\video",  # output video
        "video_size": (640, 360),  # The size of the frames of the video
        "fps": 3,  # the number of frames per second
        "method": True,  # True for the first method, False for the second method
        "img_size": 500,
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
    # Predict (for a single image)
    # =============================================
    # convert img to tensor
    img = Image.open(args.input_img)
    img = transforms.Resize((args.img_size))(img)
    # img = transforms.RandomCrop((args.img_size, args.img_size))(img)
    x = transforms.ToTensor()(img)
    x = torch.unsqueeze(x, dim=0)[:, 0:3, :, :]

    # create mask
    temp_path = "GLCIC\\test1.jpg"
    save_image(x, temp_path, nrow=1)
    if args.method:
        area = get_masked_area(temp_path)
        mask = generate_mask(
            shape=(1, 1, x.shape[2], x.shape[3]),
            area=area
        )
    else:
        area = extract_subtitle(temp_path)
        mask = generate_mask_from_pos(
            shape=(1, 1, x.shape[2], x.shape[3]),
            pos=area
        )

    # plt.imshow(mask[0][:, :].squeeze().numpy().astype(
    #     np.float32), cmap='Greys')
    # plt.show()

    # inpaint
    model.eval()
    with torch.no_grad():

        x_mask = x - x * mask + mpv * mask
        # save_image(x_mask, temp_path, nrow=1)
        input = torch.cat((x_mask, mask), dim=1)
        output = model(input)
        inpainted = poisson_blend(x_mask, output, mask)
        # imgs = torch.cat((x, x_mask, inpainted), dim=0)
        imgs = inpainted.clone()
        save_image(imgs, args.output_img, nrow=3)
    print('output img was saved as %s.' % args.output_img)

    # saliency map

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
    # os.remove(temp_path)

    area = get_masked_area(temp_path)
    x1, y, w, h = area

    img = Image.open(args.input_img2)
    img = transforms.Resize((args.img_size))(img)
    # img = transforms.RandomCrop((args.img_size, args.img_size))(img)
    x = transforms.ToTensor()(img)
    x = torch.unsqueeze(x, dim=0)[:, 0:3, :, :]
    temp_path = "GLCIC\\ssim_compare1.jpg"
    save_image(x[0, :, y: y + h, x1: x1 + w], temp_path, nrow=3)
    temp_path = "GLCIC\\ssim_compare2.jpg"
    save_image(inpainted[0, :, y: y + h, x1: x1 + w], temp_path, nrow=3)

    # ssim
    # image1 = image_convert_shape(inpainted[0, :, y: y + h, x1: x1 + w])
    # image2 = image_convert_shape(x[0, :,  y: y + h, x1: x1 + w])

    image1 = image_convert_shape(inpainted[0])
    image2 = image_convert_shape(x[0])
    print(ssim(image1, image2))

    # =============================================
    # Predict (from a set of images)
    # =============================================
    # make sure you have modify args correctly
    # make_video(args, mpv, model)
    print(psnr(image1, image2))
