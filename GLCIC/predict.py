import os
import argparse
import json
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from model import CompletionNetwork
from train import poisson_blend, generate_mask


parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('config')
parser.add_argument('input_img')
parser.add_argument('output_img')
parser.add_argument('--max_holes', type=int, default=5)
parser.add_argument('--img_size', type=int, default=160)
parser.add_argument('--hole_min_w', type=int, default=24)
parser.add_argument('--hole_max_w', type=int, default=48)
parser.add_argument('--hole_min_h', type=int, default=24)
parser.add_argument('--hole_max_h', type=int, default=48)


if __name__ == "__main__":
    args = AttrDict()
    # set hyperparameters
    args_dict = {
        "gpu": True,
        "data_dir": "./img_align_celeba/",
        "result_dir": "./results/result/",
        "data_parallel": True,
        "recursive_search": False,
        "steps_1": 1,
        "steps_2": 1,
        "steps_3": 1,
        "snaperiod_1": 10000,
        "snaperiod_2": 2000,
        "snaperiod_3": 10000,
        "max_holes": 1,
        "hole_min_w": 48,
        "hole_max_w": 96,
        "hole_min_h": 48,
        "hole_max_h": 96,
        "cn_input_size": 160,
        "ld_input_size": 96,
        "bsize": 16,
        "bdivs": 1,
        "num_test_completions": 16,
        "mpv": None,
        "alpha": 4e-4,
        "arc": 'celeba',  # 'celeba' or 'places2'
    }
    # set pretrained models if necessary
    pretrained_cn = None
    pretrained_cd = None
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
