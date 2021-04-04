"""
https://github.com/otenim/GLCIC-PyTorch/blob/master/datasets/make_dataset.py
"""
import os
import imghdr
import random
import shutil
import tqdm


def make_dataset(data_dir, split=0.8):
    """
    Make datasets for training and testing based on the images in data_dir.
    The parameter split represent the percentage of the training according to the given data.
    The defalt value is 0.8, which means 80% for training, 20% for testing.
    """
    dir = os.path.expanduser(data_dir)

    print('loading dataset...')
    src_paths = []
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if imghdr.what(path) is None:
            continue
        src_paths.append(path)
    random.shuffle(src_paths)

    # separate the paths
    border = int(split * len(src_paths))
    train_paths = src_paths[:border]
    test_paths = src_paths[border:]
    print('train images: %d images.' % len(train_paths))
    print('test images: %d images.' % len(test_paths))

    # create dst directories
    train_dir = os.path.join(dir, 'train')
    test_dir = os.path.join(dir, 'test')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # move the image files
    pbar = tqdm.tqdm(total=len(src_paths))
    for dset_paths, dset_dir in zip([train_paths, test_paths], [train_dir, test_dir]):
        for src_path in dset_paths:
            dst_path = os.path.join(dset_dir, os.path.basename(src_path))
            shutil.move(src_path, dst_path)
            pbar.update()
    pbar.close()


if __name__ == '__main__':
    # dataset directory
    data_dir = "./img_align_celeba/"
    split = 0.8
    make_dataset(data_dir, split)
