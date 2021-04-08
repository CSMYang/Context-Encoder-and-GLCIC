import os
import argparse
import json
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from model import CompletionNetwork
from train import poisson_blend, generate_mask


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


if __name__ == "__main__":
    args = AttrDict()
    # set hyperparameters
    args_dict = {
        "model": "./img_align_celeba/",
        "config": "./results/result/",
        "input_img": "", # input img location
        "output_img": "", # output img location
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
    img = transforms.RandomCrop((args.img_size, args.img_size))(img)
    x = transforms.ToTensor()(img)
    x = torch.unsqueeze(x, dim=0)

    # create mask
    mask = gen_input_mask(
        shape=(1, 1, x.shape[2], x.shape[3]),
        hole_size=(
            (args.hole_min_w, args.hole_max_w),
            (args.hole_min_h, args.hole_max_h),
        ),
        max_holes=args.max_holes,
    )

    # inpaint
    model.eval()
    with torch.no_grad():
        x_mask = x - x * mask + mpv * mask
        input = torch.cat((x_mask, mask), dim=1)
        output = model(input)
        inpainted = poisson_blend(x_mask, output, mask)
        imgs = torch.cat((x, x_mask, inpainted), dim=0)
        save_image(imgs, args.output_img, nrow=3)
    print('output img was saved as %s.' % args.output_img)
