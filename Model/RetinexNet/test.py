import os
import time
import torch
import torch.backends.cudnn as cudnn
import argparse
from glob import glob
import torchvision
from torchvision.transforms import transforms

from model import DecomNet, RelightNet
from utils import load_images


parser = argparse.ArgumentParser(description='RetinexNet args setting')

parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--gpu_idx', dest='gpu_idx', default='0', help='GPU idx')

parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='directory for checkpoints')
parser.add_argument('--save_dir', dest='save_dir', default='./test_results', help='directory for testing outputs')
parser.add_argument('--test_dir', dest='test_dir', default='/home/xuhang/dataset/mono+color/left', help='directory for testing inputs')

args = parser.parse_args()

decom_net = DecomNet()
relight_net = RelightNet()

decom_checkpoint = torch.load(args.ckpt_dir + '/decom_final.pth')
relight_checkpoint = torch.load(args.ckpt_dir + '/relight_final.pth')
decom_net.load_state_dict(decom_checkpoint)
relight_net.load_state_dict(relight_checkpoint)

if args.use_gpu:
    decom_net = decom_net.cuda()
    relight_net = relight_net.cuda()
    cudnn.benchmark = True
    cudnn.enabled = True

def test():
    decom_net.eval()
    relight_net.eval()

    test_imgs = glob(args.test_dir + '/*')
    test_imgs.sort()
    start_concat = True
    for test_img in test_imgs:
        low_img = transforms.ToTensor()(load_images(test_img))[None].permute(0, 2, 3, 1)
        if args.use_gpu:
            low_img = low_img.cuda()
        with torch.no_grad():
            lr_low, r_low, l_low = decom_net(low_img)
            l_delta = relight_net(lr_low)
        s_delta = l_delta * r_low
        if start_concat:
            r_lows = r_low
            l_lows = l_low
            s_deltas = s_delta
            start_concat = False
        else:
            r_lows = torch.concat([r_lows, r_low], dim=0)
            l_lows = torch.concat([l_lows, l_low], dim=0)
            s_deltas = torch.concat([s_deltas, s_delta], dim=0)
    torchvision.utils.save_image(r_lows.permute(0, 3, 1, 2),
                                 os.path.join(args.save_dir, time.strftime("%Y-%m-%d %H:%M:%S") + '_r_low.png'), nrow=4)
    torchvision.utils.save_image(l_lows.permute(0, 3, 1, 2),
                                 os.path.join(args.save_dir, time.strftime("%Y-%m-%d %H:%M:%S") + '_l_low.png'), nrow=4)
    torchvision.utils.save_image(s_deltas.permute(0, 3, 1, 2),
                                 os.path.join(args.save_dir, time.strftime("%Y-%m-%d %H:%M:%S") + '_s_delta.png'),
                                 nrow=4)

    # low_imgs = None
    # start_concat = True
    # for test_img in test_imgs:
    #     if start_concat:
    #         low_imgs = transforms.ToTensor()(load_images(test_img))[None]
    #         start_concat = False
    #     else:
    #         low_imgs = torch.concat([low_imgs, transforms.ToTensor()(load_images(test_img))[None]], dim=0)
    # low_imgs = low_imgs.permute(0, 2, 3, 1)
    # if args.use_gpu:
    #     low_imgs = low_imgs.cuda()
    # with torch.no_grad():
    #     lr_low, r_low, l_low = decom_net(low_imgs)
    #     l_delta = relight_net(lr_low)
    # s_delta = l_delta * r_low
    # torchvision.utils.save_image(r_low.permute(0, 3, 1, 2),
    #                              os.path.join(args.save_dir, time.strftime("%Y-%m-%d %H:%M:%S") + '_r_low.png'), nrow=4)
    # torchvision.utils.save_image(l_low.permute(0, 3, 1, 2),
    #                              os.path.join(args.save_dir, time.strftime("%Y-%m-%d %H:%M:%S") + '_l_low.png'), nrow=4)
    # torchvision.utils.save_image(s_delta.permute(0, 3, 1, 2),
    #                              os.path.join(args.save_dir, time.strftime("%Y-%m-%d %H:%M:%S") + '_s_delta.png'), nrow=4)
    # TODO 貌似checkpoint未加载成功，R还是0.5左右

if __name__ == '__main__':
    test()