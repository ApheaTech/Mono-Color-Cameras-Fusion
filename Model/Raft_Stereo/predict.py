import torch
import torch.nn as nn
import numpy as np
import argparse
import torchvision.transforms.functional as TF

from Model.Raft_Stereo.raft_stereo import RAFTStereo

def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--restore_ckpt',
                        default="/home/xuhang/Mono-Color-Cameras-Fusion/Model/Raft_Stereo/checkpoint/raftstereo-middlebury.pth",
                        help="restore checkpoint")

    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3,
                        help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg",
                        help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true',
                        help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    args = parser.parse_args()

    return args


def stereo_matching(left_gray_img, right_mono_img):

    args = getargs()
    args.mixed_precision = True
    args.slow_fast_gru = True
    args.device = 'cuda:1'

    left_gray_img_list = [left_gray_img, TF.rotate(right_mono_img, 180)]
    right_mono_img = right_mono_img[0, 0, None, None].repeat(1, 3, 1, 1)
    right_mono_img_list = [right_mono_img, TF.rotate(left_gray_img, 180)]

    model = nn.DataParallel(RAFTStereo(args), device_ids=[0])
    # model = RAFTStereo(args)
    model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model = model.module
    model.to(args.device)
    model.eval()

    disp_list = []

    for i in range(2):
        with torch.no_grad():
            image1 = left_gray_img_list[i].to(args.device) * 255
            image2 = right_mono_img_list[i].to(args.device) * 255

            _, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
            disp = -flow_up.cpu().numpy().squeeze().astype('uint16')
            np.rot90(np.rot90(disp))
            if i == 0:
                disp_list.append(disp)
            else:
                disp_list.append(np.rot90(np.rot90(disp)))

    # disp_list[0] 左视差图
    # disp_list[1] 右视差图
    return disp_list
