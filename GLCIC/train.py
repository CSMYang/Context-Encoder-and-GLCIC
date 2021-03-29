"""
https://github.com/otenim/GLCIC-PyTorch/blob/master/train.py
"""
import json
import os
import random
from torch.utils.data import DataLoader
from torch.optim import Adadelta
from torch.nn.functional import mse_loss
from torchvision.utils import save_image
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
from tqdm import tqdm
from model import CompletionNetwork, ContextDiscriminator
from dataset import ImageDataset


def gen_input_mask(shape, hole_size, hole_area=None, max_holes=1):
    """
    * inputs:
        - shape (sequence, required):
                Shape of a mask tensor to be generated.
                A sequence of length 4 (N, C, H, W) is assumed.
        - hole_size (sequence or int, required):
                Size of holes created in a mask.
                If a sequence of length 4 is provided,
                holes of size (W, H) = (
                    hole_size[0][0] <= hole_size[0][1],
                    hole_size[1][0] <= hole_size[1][1],
                ) are generated.
                All the pixel values within holes are filled with 1.0.
        - hole_area (sequence, optional):
                This argument constraints the area where holes are generated.
                hole_area[0] is the left corner (X, Y) of the area,
                while hole_area[1] is its width and height (W, H).
                This area is used as the input region of Local discriminator.
                The default value is None.
        - max_holes (int, optional):
                This argument specifies how many holes are generated.
                The default value is 1.
    * returns:
            A mask tensor of shape [N, C, H, W] with holes.
            All the pixel values within holes are filled with 1.0,
            while the other pixel values are zeros.
    """
    mask = torch.zeros(shape)
    for i in range(mask.shape[0]):
        n_holes = random.randint(1, max_holes)
        for _ in range(n_holes):
            hole_w = random.randint(hole_size[0][0], hole_size[0][1])
            hole_h = random.randint(hole_size[1][0], hole_size[1][1])
            # choose offset upper-left coordinate
            if hole_area:
                harea_xmin, harea_ymin = hole_area[0]
                harea_w, harea_h = hole_area[1]
                offset_x = random.randint(harea_xmin, harea_xmin + harea_w - hole_w)
                offset_y = random.randint(harea_ymin, harea_ymin + harea_h - hole_h)
            else:
                offset_x = random.randint(0, mask.shape[3] - hole_w)
                offset_y = random.randint(0, mask.shape[2] - hole_h)
            mask[i, :, offset_y: offset_y + hole_h, offset_x: offset_x + hole_w] = 1.0
    return mask


def gen_hole_area(size, mask_size):
    """
    * inputs:
        - size (sequence, required)
                A sequence of length 2 (W, H) is assumed.
                (W, H) is the size of hole area.
        - mask_size (sequence, required)
                A sequence of length 2 (W, H) is assumed.
                (W, H) is the size of input mask.
    * returns:
            A sequence used for the input argument 'hole_area' for function 'gen_input_mask'.
    """
    mask_w, mask_h = mask_size
    harea_w, harea_h = size
    offset_x = random.randint(0, mask_w - harea_w)
    offset_y = random.randint(0, mask_h - harea_h)
    return ((offset_x, offset_y), (harea_w, harea_h))


def crop(x, area):
    """
    * inputs:
        - x (torch.Tensor, required)
                A torch tensor of shape (N, C, H, W) is assumed.
        - area (sequence, required)
                A sequence of length 2 ((X, Y), (W, H)) is assumed.
                sequence[0] (X, Y) is the left corner of an area to be cropped.
                sequence[1] (W, H) is its width and height.
    * returns:
            A torch tensor of shape (N, C, H, W) cropped in the specified area.
    """
    xmin, ymin = area[0]
    w, h = area[1]
    return x[:, :, ymin: ymin + h, xmin: xmin + w]


def sample_random_batch(dataset, batch_size=32):
    """
    * inputs:
        - dataset (torch.utils.data.Dataset, required)
                An instance of torch.utils.data.Dataset.
        - batch_size (int, optional)
                Batch size.
    * returns:
            A mini-batch randomly sampled from the input dataset.
    """
    num_samples = len(dataset)
    batch = []
    for _ in range(min(batch_size, num_samples)):
        index = random.randint(0, num_samples-1)
        x = torch.unsqueeze(dataset[index], dim=0)
        batch.append(x)
    return torch.cat(batch, dim=0)


def poisson_blend(input, output, mask):
    """
    * inputs:
        - input (torch.Tensor, required)
                Input tensor of Completion Network, whose shape = (N, 3, H, W).
        - output (torch.Tensor, required)
                Output tensor of Completion Network, whose shape = (N, 3, H, W).
        - mask (torch.Tensor, required)
                Input mask tensor of Completion Network, whose shape = (N, 1, H, W).
    * returns:
                Output image tensor of shape (N, 3, H, W) inpainted with poisson image editing method.
    """
    input, output, mask = input.clone(), output.clone(), mask.clone()
    mask = torch.cat((mask, mask, mask), dim=1)  # convert to 3-channel format
    ret = []
    for i in range(input.shape[0]):
        dst_img = np.array(to_pil_image(input[i]))[:, :, [2, 1, 0]]
        srcimg = np.array(to_pil_image(output[i]))[:, :, [2, 1, 0]]
        msk = np.array(to_pil_image(mask[i]))[:, :, [2, 1, 0]]
        # compute mask's center
        xs, ys = [], []
        for j in range(msk.shape[0]):
            for k in range(msk.shape[1]):
                if msk[j, k, 0] == 255:
                    ys.append(j)
                    xs.append(k)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        center = ((xmax + xmin) // 2, (ymax + ymin) // 2)
        dst_img = cv2.inpaint(dst_img, msk[:, :, 0], 1, cv2.INPAINT_TELEA)
        out = to_tensor(cv2.seamlessClone(srcimg, dst_img, msk, center,
                                cv2.NORMAL_CLONE)[:, :, [2, 1, 0]])
        out = torch.unsqueeze(out, dim=0)
        ret.append(out)
    ret = torch.cat(ret, dim=0)
    return ret


def completion_network_loss(input, output, mask):
    return mse_loss(output * mask, input * mask)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def train_p1(args, pretrained_cn, train_loader, mpv, test_set):
    """
    Trainning Phase 1: train completion network
    """

    # load completion network
    model_cn = CompletionNetwork()
    # load pre-trained model
    if pretrained_cn:
        model_cn.load_state_dict(torch.load(args.pretrained_cn))
    if args.data_parallel:
        model_cn = torch.nn.DataParallel(model_cn)
    if args.gpu:
        model_cn = model_cn.cuda()
    opt_cn = Adadelta(model_cn.parameters())

    # training
    cnt_bdivs = 0
    pbar = tqdm(total=args.steps_1)
    while pbar.n < args.steps_1:
        for x in train_loader:
            # forward
            mask = gen_input_mask(shape=(x.shape[0], 1, x.shape[2], x.shape[3]),
                                  hole_size=((args.hole_min_w, args.hole_max_w),
                                             (args.hole_min_h, args.hole_max_h)),
                                  hole_area=gen_hole_area((args.ld_input_size,
                                                           args.ld_input_size),
                                                          (x.shape[3], x.shape[2])),
                                  max_holes=args.max_holes)
            if args.gpu:
                x = x.cuda()
                mask = mask.cuda()
            x_mask = x - x * mask + mpv * mask
            input = torch.cat((x_mask, mask), dim=1)
            output = model_cn(input)
            loss = completion_network_loss(x, output, mask)

            # backward
            loss.backward()
            cnt_bdivs += 1
            if cnt_bdivs >= args.bdivs:
                cnt_bdivs = 0

                # optimize
                opt_cn.step()
                opt_cn.zero_grad()
                pbar.set_description('phase 1 | train loss: %.5f' % loss)
                pbar.update()

                # test
                if pbar.n % args.snaperiod_1 == 0:
                    model_cn.eval()
                    x = sample_random_batch(test_set,
                                            batch_size=args.num_test_completions)
                    mask = gen_input_mask(shape=(x.shape[0], 1, x.shape[2], x.shape[3]),
                                          hole_size=((args.hole_min_w, args.hole_max_w),
                                                     (args.hole_min_h, args.hole_max_h)),
                                          hole_area=gen_hole_area((args.ld_input_size, args.ld_input_size),
                                                                  (x.shape[3], x.shape[2])),
                                          max_holes=args.max_holes)
                    if args.gpu:
                        x = x.cuda()
                        mask = mask.cuda()
                    x_mask = x - x * mask + mpv * mask
                    input = torch.cat((x_mask, mask), dim=1)
                    with torch.no_grad():
                        output = model_cn(input)
                    completed = poisson_blend(x_mask, output, mask)
                    imgs = torch.cat((x, x_mask, completed), dim=0)
                    if args.gpu:
                        imgs = imgs.cuda()
                    imgpath = os.path.join(args.result_dir, 'phase_1', 'step%d.png' % pbar.n)
                    model_cn_path = os.path.join(args.result_dir, 'phase_1', 'model_cn_step%d' % pbar.n)
                    save_image(imgs, imgpath, nrow=len(x))
                    if args.data_parallel:
                        torch.save(model_cn.module.state_dict(), model_cn_path)
                    else:
                        torch.save(model_cn.state_dict(), model_cn_path)
                    model_cn.train()
                if pbar.n >= args.steps_1:
                    break
    pbar.close()
    return model_cn, opt_cn


def train_p2(args, pretrained_cd, train_loader, mpv, test_set, model_cn):
    """
    Training Phase 2: train context discriminator
    """
    # load context discriminator
    model_cd = ContextDiscriminator(local_in_shape=(3, args.ld_input_size, args.ld_input_size),
                                    global_in_shape=(3, args.cn_input_size, args.cn_input_size),
                                    architecture=args.arc)
    if pretrained_cd:
        model_cd.load_state_dict(torch.load(args.pretrained_cd))
    if args.data_parallel:
        model_cd = torch.nn.DataParallel(model_cd)
    if args.gpu:
        model_cd = model_cd.cuda()
    opt_cd = Adadelta(model_cd.parameters())
    bceloss = torch.nn.BCELoss()

    # training
    cnt_bdivs = 0
    pbar = tqdm(total=args.steps_2)
    while pbar.n < args.steps_2:
        for x in train_loader:
            # fake forward
            hole_area_fake = gen_hole_area((args.ld_input_size, args.ld_input_size),
                                           (x.shape[3], x.shape[2]))
            mask = gen_input_mask(shape=(x.shape[0], 1, x.shape[2], x.shape[3]),
                                  hole_size=((args.hole_min_w, args.hole_max_w),
                                             (args.hole_min_h, args.hole_max_h)),
                                  hole_area=hole_area_fake, max_holes=args.max_holes)
            fake = torch.zeros((len(x), 1))
            if args.gpu:
                x = x.cuda()
                mask = mask.cuda()
                fake = fake.cuda()
            x_mask = x - x * mask + mpv * mask
            input_cn = torch.cat((x_mask, mask), dim=1)
            output_cn = model_cn(input_cn)
            input_gd_fake = output_cn.detach()
            input_ld_fake = crop(input_gd_fake, hole_area_fake)
            if args.gpu:
                input_gd_fake = input_gd_fake.cuda()
                input_ld_fake = input_ld_fake.cuda()
            output_fake = model_cd((input_ld_fake, input_gd_fake))
            loss_fake = bceloss(output_fake, fake)

            # real forward
            hole_area_real = gen_hole_area((args.ld_input_size, args.ld_input_size),
                                           (x.shape[3], x.shape[2]))
            real = torch.ones((len(x), 1))
            if args.gpu:
                real = real.cuda()
            input_gd_real = x
            input_ld_real = crop(input_gd_real, hole_area_real)
            output_real = model_cd((input_ld_real, input_gd_real))
            loss_real = bceloss(output_real, real)

            # reduce
            loss = (loss_fake + loss_real) / 2.

            # backward
            loss.backward()
            cnt_bdivs += 1
            if cnt_bdivs >= args.bdivs:
                cnt_bdivs = 0

                # optimize
                opt_cd.step()
                opt_cd.zero_grad()
                pbar.set_description('phase 2 | train loss: %.5f' % loss)
                pbar.update()

                # test
                if pbar.n % args.snaperiod_2 == 0:
                    model_cn.eval()
                    with torch.no_grad():
                        x = sample_random_batch(test_set, batch_size=args.num_test_completions)
                        mask = gen_input_mask(shape=(x.shape[0], 1, x.shape[2], x.shape[3]),
                                              hole_size=((args.hole_min_w, args.hole_max_w),
                                                         (args.hole_min_h, args.hole_max_h)),
                                              hole_area=gen_hole_area((args.ld_input_size, args.ld_input_size),
                                                                      (x.shape[3], x.shape[2])),
                                              max_holes=args.max_holes)
                        if args.gpu:
                            x = x.cuda()
                            mask = mask.cuda()
                        x_mask = x - x * mask + mpv * mask
                        input = torch.cat((x_mask, mask), dim=1)
                        output = model_cn(input)
                        completed = poisson_blend(x_mask, output, mask)
                        imgs = torch.cat((x, x_mask, completed), dim=0)
                        if args.gpu:
                            imgs = imgs.cuda()
                        imgpath = os.path.join(args.result_dir, 'phase_2', 'step%d.png' % pbar.n)
                        model_cd_path = os.path.join(args.result_dir, 'phase_2', 'model_cd_step%d' % pbar.n)
                        save_image(imgs, imgpath, nrow=len(x))
                        if args.data_parallel:
                            torch.save(model_cd.module.state_dict(), model_cd_path)
                        else:
                            torch.save(model_cd.state_dict(), model_cd_path)
                    model_cn.train()
                if pbar.n >= args.steps_2:
                    break
    pbar.close()
    return model_cd, opt_cd


def train(args, pretrained_cn, pretrained_cd):
    # ================================================
    # Preparation
    # ================================================

    # create result directory as needed
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    for phase in ['phase_1', 'phase_2', 'phase_3']:
        if not os.path.exists(os.path.join(args.result_dir, phase)):
            os.makedirs(os.path.join(args.result_dir, phase))

    # load dataset
    transformed = transforms.Compose([
        transforms.Resize(args.cn_input_size),
        transforms.RandomCrop((args.cn_input_size, args.cn_input_size)),
        transforms.ToTensor(),
    ])
    print('loading dataset...')
    train_set = ImageDataset(os.path.join(args.data_dir, 'train'), transformed,
                             recursive_search=args.recursive_search)
    test_set = ImageDataset(os.path.join(args.data_dir, 'test'), transformed,
                            recursive_search=args.recursive_search)
    train_loader = DataLoader(train_set, batch_size=(args.bsize // args.bdivs),
                              shuffle=True)

    # compute mpv (mean pixel value) of training dataset
    mpv = np.zeros(shape=(3,))
    pbar = tqdm(total=len(train_set.images),
                desc='computing mean pixel value of training dataset...')
    for imgpath in train_set.images:
        img = Image.open(imgpath)
        x = np.array(img) / 255.
        mpv += x.mean(axis=(0, 1))
        pbar.update()
    mpv /= len(train_set.images)
    pbar.close()
    if args.mpv:
        mpv = np.array(args.mpv)

    # save training config
    mpv_json = []
    for i in range(3):
        mpv_json.append(float(mpv[i]))
    args_dict = vars(args)
    args_dict['mpv'] = mpv_json
    with open(os.path.join(args.result_dir, 'config.json'), mode='w') as f:
        json.dump(args_dict, f)

    # make mpv & alpha tensors
    mpv = mpv.reshape(1, 3, 1, 1).float()
    alpha = torch.tensor(args.alpha, dtype=torch.float32)

    if args.gpu:
        mpv = mpv.cuda()
        alpha = alpha.cuda()

    # Phase 1
    model_cn, opt_cn = train_p1(args, pretrained_cn, train_loader, mpv, test_set)

    # Phase 2
    model_cd, opt_cd = train_p2(args, pretrained_cd, train_loader, mpv, test_set, model_cn)

    # ================================================
    # Training Phase 3
    # ================================================
    cnt_bdivs = 0
    pbar = tqdm(total=args.steps_3)
    bceloss = torch.nn.BCELoss()
    while pbar.n < args.steps_3:
        for x in train_loader:
            # forward model_cd
            hole_area_fake = gen_hole_area((args.ld_input_size, args.ld_input_size),
                                           (x.shape[3], x.shape[2]))
            mask = gen_input_mask(shape=(x.shape[0], 1, x.shape[2], x.shape[3]),
                                  hole_size=((args.hole_min_w, args.hole_max_w),
                                             (args.hole_min_h, args.hole_max_h)),
                                  hole_area=hole_area_fake, max_holes=args.max_holes)

            # fake forward
            fake = torch.zeros((len(x), 1))
            if args.gpu:
                x = x.cuda()
                mask = mask.cuda()
                fake = fake.cuda()
            x_mask = x - x * mask + mpv * mask
            input_cn = torch.cat((x_mask, mask), dim=1)
            output_cn = model_cn(input_cn)
            input_gd_fake = output_cn.detach()
            input_ld_fake = crop(input_gd_fake, hole_area_fake)
            output_fake = model_cd((input_ld_fake, input_gd_fake))
            loss_cd_fake = bceloss(output_fake, fake)

            # real forward
            hole_area_real = gen_hole_area((args.ld_input_size, args.ld_input_size),
                                           (x.shape[3], x.shape[2]))
            real = torch.ones((len(x), 1))
            if args.gpu:
                real = real.cuda()
            input_gd_real = x
            input_ld_real = crop(input_gd_real, hole_area_real)
            output_real = model_cd((input_ld_real, input_gd_real))
            loss_cd_real = bceloss(output_real, real)

            # reduce
            loss_cd = (loss_cd_fake + loss_cd_real) * alpha / 2.

            # backward model_cd
            loss_cd.backward()
            cnt_bdivs += 1
            if cnt_bdivs >= args.bdivs:
                # optimize
                opt_cd.step()
                opt_cd.zero_grad()

            # forward model_cn
            loss_cn_1 = completion_network_loss(x, output_cn, mask)
            input_gd_fake = output_cn
            input_ld_fake = crop(input_gd_fake, hole_area_fake)
            output_fake = model_cd((input_ld_fake, input_gd_fake))
            loss_cn_2 = bceloss(output_fake, real)

            # reduce
            loss_cn = (loss_cn_1 + alpha * loss_cn_2) / 2.

            # backward model_cn
            loss_cn.backward()
            if cnt_bdivs >= args.bdivs:
                cnt_bdivs = 0

                # optimize
                opt_cn.step()
                opt_cn.zero_grad()
                pbar.set_description('phase 3 | train loss (cd): %.5f (cn): %.5f' % (loss_cd, loss_cn))
                pbar.update()

                # test
                if pbar.n % args.snaperiod_3 == 0:
                    model_cn.eval()
                    x = sample_random_batch(test_set, batch_size=args.num_test_completions)
                    mask = gen_input_mask(shape=(x.shape[0], 1, x.shape[2], x.shape[3]),
                                          hole_size=((args.hole_min_w, args.hole_max_w),
                                                     (args.hole_min_h, args.hole_max_h)),
                                          hole_area=gen_hole_area((args.ld_input_size, args.ld_input_size),
                                                                  (x.shape[3], x.shape[2])),
                                          max_holes=args.max_holes)
                    if args.gpu:
                        x = x.cuda()
                        mask = mask.cuda()
                    x_mask = x - x * mask + mpv * mask
                    input = torch.cat((x_mask, mask), dim=1)
                    with torch.no_grad():
                        output = model_cn(input)
                    completed = poisson_blend(x_mask, output, mask)
                    if args.gpu:
                        completed = completed.cuda()
                    imgs = torch.cat((x, x_mask, completed), dim=0)
                    if args.gpu:
                        imgs = imgs.cuda()
                    imgpath = os.path.join(args.result_dir, 'phase_3', 'step%d.png' % pbar.n)
                    model_cn_path = os.path.join(args.result_dir, 'phase_3', 'model_cn_step%d' % pbar.n)
                    model_cd_path = os.path.join(args.result_dir, 'phase_3', 'model_cd_step%d' % pbar.n)
                    save_image(imgs, imgpath, nrow=len(x))
                    if args.data_parallel:
                        torch.save(model_cn.module.state_dict(), model_cn_path)
                        torch.save(model_cd.module.state_dict(), model_cd_path)
                    else:
                        torch.save(model_cn.state_dict(), model_cn_path)
                        torch.save(model_cd.state_dict(), model_cd_path)
                    model_cn.train()
                if pbar.n >= args.steps_3:
                    break
    pbar.close()


if __name__ == '__main__':
    args = AttrDict()
    # set hyperparameters
    args_dict = {
        "gpu": True,
        "data_dir": "./datasets/dataset/",
        "result_dir": "./results/result/",
        "data_parallel": True,
        "recursive_search": False,
        "steps_1": 90000,
        "steps_2": 10000,
        "steps_3": 400000,
        "snaperiod_1": 10000,
        "snaperiod_2": 2000,
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
    train(args, pretrained_cn, pretrained_cd)
