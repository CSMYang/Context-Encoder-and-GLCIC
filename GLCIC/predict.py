import json
import os
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from model_pretrained import CompletionNetwork
from train import poisson_blend, generate_mask
from dataset import ImageDataset


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def test_images(args, mpv, model):
    """
    Predict imgs from dir
    """
    # convert img to tensor
    transformed = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.RandomCrop((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])
    image_set = ImageDataset(os.path.join(args.data_dir, 'train'), transformed)
    mask_path = args.output_img + "/mask"
    output_path = args.output_img + "/output"

    for i in range(image_set):
        x = torch.unsqueeze(image_set[i], dim=0)

        # create mask
        mask = generate_mask(
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
            maskpath = os.path.join(mask_path, 'test_%d.png' % i)
            outputpath = os.path.join(output_path, 'test_%d.png' % i)
            save_image(imgs, maskpath)
            save_image(inpainted, outputpath)
    print('output img was saved as %s.' % args.output_img)


if __name__ == "__main__":
    args = AttrDict()
    # set hyperparameters
    args_dict = {
        "model": "GLCIC\pretrained_model_cn",
        "config": "GLCIC\config.json",
        "input_img": "img_align_celeba\\test\\000003.jpg",  # input img
        "output_img": "GLCIC\\result.jpg",  # output img name
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
    mask = generate_mask(
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
