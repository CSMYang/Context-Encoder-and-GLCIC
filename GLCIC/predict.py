import json
import os
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from model_pretrained import CompletionNetwork
from train import poisson_blend, generate_mask
from dataset import ImageDataset
from evaluation import *
# from detect_subtitle.py import *


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
        transforms.Resize((args.img_size, args.img_size)),
        # transforms.RandomCrop((args.img_size, args.img_size)),
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
        "input_img": "GLCIC\movie_caption.jpg",  # input img
        "output_img": "GLCIC\\result.jpg",  # output img name
        "max_holes": 10,
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
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()

    ])

    train_set = ImageDataset("GLCIC\TestResultForFid\Real",
                             transform, recursive_search=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                               shuffle=False)
    total_psnr = 0
    total_ssim = 0
    for i, img in enumerate(train_loader):
        # img = Image.open(args.input_img)
        x = (img)
        x = x[:, 0: 3, :, :]

        # create mask
        mask = generate_mask(
            shape=(1, 1, x.shape[2], x.shape[3]),
            hole_size=(
                (args.hole_min_w, args.hole_max_w),
                (args.hole_min_h, args.hole_max_h),
            ),
            max_holes=args.max_holes,
        )
        # center masking

        center_mask = torch.zeros((1, 1, x.shape[2], x.shape[3]))
        center_mask[:, :, 80-32:80+32, 80-32:80+32] = 1

        # img_path = "GLCIC\caption_masked.jpg"
        # area = get_area(img_path)
        # image = cv2.imread(img_path)
        # mask = torch.Tensor(generate_mask(image.shape, area))

        mask = center_mask
        # inpaint
        model.eval()
        with torch.no_grad():
            x_mask = x - x * mask + mpv * mask
            input = torch.cat((x_mask, mask), dim=1)
            output = model(input)
            inpainted = poisson_blend(x_mask, output, mask)
            # imgs = torch.cat((x, x_mask, inpainted), dim=0)
            imgs = inpainted
            save_image(
                imgs, "GLCIC\TestResultForFid\\Fake\\{}_generated.png".format(i), nrow=3)
        # print(ssim(result.cpu().detach(), temp_temp_image_1))

        inpainted = image_convert_shape(inpainted[0])
        x = image_convert_shape(x[0])
        ssim_result = ssim(
            inpainted[80-32:80+32, 80-32:80+32, :].cpu().detach(), x[80-32:80+32, 80-32:80+32, :])

        psnr_result = psnr(
            inpainted[80-32:80+32, 80-32:80+32, :].cpu().detach(), x[80-32:80+32, 80-32:80+32, :])
        total_psnr += psnr_result
        total_ssim += ssim_result

        print("{}".format(i), "ssim :", ssim_result, "psnr:", psnr_result)
    print("ssim :", total_ssim/10, "psnr:", total_psnr/10)
