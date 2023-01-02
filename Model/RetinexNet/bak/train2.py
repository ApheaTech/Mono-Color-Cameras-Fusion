import torch

import torchvision.utils
from torch.utils.tensorboard import SummaryWriter

from model import DecomNet, RelightNet
from loss import DecomLoss, RelightLoss
import tqdm

from Model.RetinexNet.bak import MyLoader as DA

# 数据集准备
# all_low_img, all_high_img = lt.dataloader("/home/xuhang/dataset/LOLdataset")
TrainImgLoader = torch.utils.data.DataLoader(DA.myImageFolder("/home/xuhang/dataset/LOLdataset", training=True),
                                             batch_size=1, shuffle=True, num_workers=8, drop_last=False)
EvalImgLoader = torch.utils.data.DataLoader(DA.myImageFolder("/home/xuhang/dataset/LOLdataset", training=False),
                                             batch_size=1, shuffle=False, num_workers=8, drop_last=False)

# 模型准备
DEVICE = 'cuda:1'
decom_net = DecomNet().to(DEVICE)
relight_net = RelightNet().to(DEVICE)
decom_optim = torch.optim.Adam(decom_net.parameters(), lr=0.0001)
relight_optim = torch.optim.Adam(relight_net.parameters(), lr=0.0001)
decom_criterion = DecomLoss()
relight_criterion = RelightLoss()

# 日志准备
writer = SummaryWriter("./checkpoint/log")

def train():

    for epoch in range(30):

        # 训练decom_net
        decom_net.train()
        times_per_epoch, sum_loss = 0, 0.
        for batch in tqdm.tqdm(TrainImgLoader):

            times_per_epoch += 1
            low_im, high_im = [tensor.to(DEVICE) for tensor in batch]

            decom_optim.zero_grad()

            _, r_low, l_low = decom_net(low_im)
            _, r_high, l_high = decom_net(high_im)

            loss = decom_criterion(r_low.detach().cpu(), l_low.detach().cpu(),
                                   r_high.detach().cpu(), l_high.detach().cpu(),
                                   low_im.detach().cpu(), high_im.detach().cpu()).requires_grad_()
            loss.backward()
            decom_optim.step()

            sum_loss += loss.data
            # print('loss:', loss.data)

        print('epoch: ' + str(epoch) + ' | loss: ' + str(sum_loss / times_per_epoch))
        if (epoch + 1) % 5 == 0:
            torch.save(decom_net.state_dict(), './checkpoint/decom_' + str(epoch) + '.pth')

        # 验证decom_net
        decom_net.eval()
        idx = 0
        for batch in tqdm.tqdm(EvalImgLoader):
            idx += 1
            low_im, high_im = [tensor.to(DEVICE) for tensor in batch]
            with torch.no_grad():
                _, r_low, l_low = decom_net(low_im)
                _, r_high, l_high = decom_net(high_im)
            torchvision.utils.save_image(r_low[0].permute(2, 0, 1), './result/r_low_' + str(idx) + '.png')
            torchvision.utils.save_image(l_low[0].permute(2, 0, 1), './result/l_low' + str(idx) + '.png')
            torchvision.utils.save_image(r_high[0].permute(2, 0, 1), './result/r_high' + str(idx) + '.png')
            torchvision.utils.save_image(l_high[0].permute(2, 0, 1), './result/l_high' + str(idx) + '.png')
            # writer.add_image('r_low', r_low[0].permute(2, 0, 1), idx)
            # writer.add_image('l_low', l_low[0].permute(2, 0, 1), idx)
            # writer.add_image('r_high', r_high[0].permute(2, 0, 1), idx)
            # writer.add_image('l_high', l_high[0].permute(2, 0, 1), idx)

    torch.save(decom_net.state_dict(), '../checkpoint/decom_final.pth')

    # 训练relight_net
    for epoch in range(30):
        relight_net.train()
        times_per_epoch, sum_loss = 0, 0.
        for batch in tqdm.tqdm(TrainImgLoader):
            low_im, high_im = [tensor.to(DEVICE) for tensor in batch]

            relight_optim.zero_grad()

            lr_low, r_low, _ = decom_net(low_im)
            l_delta = relight_net(lr_low)

            loss = relight_criterion(l_delta.detach().cpu(),
                                     r_low.detach().cpu(),
                                     high_im.detach().cpu()).requires_grad_()
            loss.backward()
            relight_optim.step()

        print('epoch: ' + str(epoch) + ' | loss: ' + str(sum_loss / times_per_epoch))
        if (epoch + 1) % 5 == 0:
            torch.save(relight_net.state_dict(), './checkpoint/relight_' + str(epoch) + '.pth')
    torch.save(relight_net.state_dict(), '../checkpoint/relight_final.pth')


if __name__ == '__main__':

    train()
