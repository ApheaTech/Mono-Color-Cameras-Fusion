import os
import torch.utils.data as data
from PIL import Image, ImageOps
import random
import numpy as np
import torchvision.transforms as transforms

def rgb_loader(path):
    im = Image.open(path)
    # np.array(im, dtype="float32") / 255.0
    return transforms.ToTensor()(im).permute(1, 2, 0)

def ycbcr_loader(path):
    return Image.open(path).convert('YCbCr')

def gray_loader(path):
    return Image.open(path).convert('L')

def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counter-wise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)

# 数据集结构
class myImageFolder(data.Dataset):
    def __init__(self, rootpath, training):
        self.low_img = []
        self.high_img = []
        if training:
            filepath = os.path.join(rootpath, 'our485')
        else:
            filepath = os.path.join(rootpath, 'eval15')
        low_path = os.path.join(filepath, 'low')
        high_path = os.path.join(filepath, 'high')
        for img_name in os.listdir(low_path):
            self.low_img.append(os.path.join(low_path, img_name))
            self.high_img.append(os.path.join(high_path, img_name))

    def __getitem__(self, item):
        low_img = rgb_loader(self.low_img[item])
        high_img = rgb_loader(self.high_img[item])

        # h, w, _ = low_img.shape
        # patch_size = 48
        # x = random.randint(0, h - patch_size)
        # y = random.randint(0, w - patch_size)
        # rand_mode = random.randint(0, 7)
        # low_img = data_augmentation(low_img[x:x + patch_size, y:y + patch_size, :], rand_mode)
        # high_img = data_augmentation(high_img[x:x + patch_size, y:y + patch_size, :], rand_mode)

        return low_img, high_img

    def __len__(self):
        return len(self.low_img)
