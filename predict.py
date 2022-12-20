import torch
import numpy as np
import time
from PIL import Image
from torchvision.transforms import transforms

from Utils.utils import InputPadder
from Model.Raft_Stereo.predict import stereo_matching


def read_img_tensor(left_img_path, right_img_path):
    left_color_img = transforms.ToTensor()(Image.open(left_img_path).convert('YCbCr')).unsqueeze(0)
    left_gray_img = left_color_img[0, 0].unsqueeze(0).unsqueeze(1).repeat(1, 3, 1, 1)
    right_mono_img = transforms.ToTensor()(Image.open(right_img_path).convert('YCbCr')).unsqueeze(0)
    padder = InputPadder((left_color_img.shape[2], left_color_img.shape[3]), divis_by=32)
    left_color_img, right_mono_img, left_gray_img = padder.pad(left_color_img, right_mono_img, left_gray_img)

    return left_color_img, right_mono_img, left_gray_img


def fusion(disparity, left_color_img, right_mono_img):
    left_color_img = transforms.ToPILImage()(left_color_img.squeeze(0))
    right_mono_img = transforms.ToPILImage()(right_mono_img.squeeze(0))
    left_img_ycbcr = np.array(left_color_img).astype('uint8')
    right_img_ycbcr = np.array(right_mono_img).astype('uint8')

    w = disparity.shape[-1]
    coord = np.linspace(0, w - 1, w)[None,]  # 1xW
    right_shifted = coord - disparity  # right_shifted是左图中像素变换到右图中的位置
    occ_mask_l = right_shifted <= 0
    right_shifted[occ_mask_l] = 0  # set negative locations to 0
    right_shifted = right_shifted.astype('int16')  # 左图像像素在右图像中的同名点的纵坐标

    # 扭曲的彩色图像
    left_shift = np.zeros_like(left_img_ycbcr)
    # 融合图像
    for i in range(right_img_ycbcr.shape[0]):
        left_shift[i, right_shifted[i, :], :] = left_img_ycbcr[i, :, :]
    a = left_shift[:, :, 1:3]
    a[a == 0] = 128

    right_img_ycbcr[:, :, 1:3] = a[:, :, 0:2]

    right_img_ycbcr = torch.tensor(right_img_ycbcr).permute(2, 0, 1) / 255
    left_shift = torch.tensor(left_shift).permute(2, 0, 1) / 255

    return right_img_ycbcr, left_shift


def main():

    left_img_path = '/home/xuhang/dataset/P10_1123_enhance/view0.png'
    right_img_path = '/home/xuhang/dataset/P10_1123_enhance/view2.png'

    # 读取待融合图片
    left_color_img, right_mono_img, left_gray_img = read_img_tensor(left_img_path, right_img_path)

    # 图像匹配
    print('Start image matching...')
    start = time.time()
    disp_list = stereo_matching(left_gray_img, right_mono_img)
    print("Compelete image match used time %.2f" % (time.time()-start))

    # 空间域融合
    # TODO 使用grid_sample实现
    print('Start image fusion...')
    fusion_img, left_warp = fusion(disp_list[0], left_color_img, right_mono_img)

    # 保存
    x = transforms.ToPILImage(mode='YCbCr')(fusion_img).convert('RGB')
    x.save('fusion.png')

    print('Complete!')


if __name__ == '__main__':
    main()
