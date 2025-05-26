import os
import time
import albumentations as A
import numpy as np
import torch
import torch.backends.cudnn
import torch.utils.data
import yaml
from attrdict import AttrMap
from torch import nn
from torch.utils.data import DataLoader
import pytorch_ssim
from modules.BasicBlock import oneStepLoss, fftLoss, ContrastLoss
from modules.ACA import TMnet
#from modules.UformerNet import Ufnet as TMnet#使用former
#from modules.dehazeformer import TMnet
from pytorch_msssim import MS_SSIM
from utils.datasets import TrainDataset, tDataset
from utils.lossReport import lossReport, TestReport
from utils.utils import gpu_manage, checkpoint, adjust_learning_rate, print_model, rfft
from val import val
from torchsummary import summary
ssim_loss = pytorch_ssim.SSIM()
# restLoss = TwoStepLoss()
preloss = oneStepLoss()
criterion_fft = fftLoss()
crloss = ContrastLoss()
MSSSIM = MS_SSIM(data_range=1., size_average=True, channel=1).cuda()

import cv2

trfm = A.Compose([
    A.Rotate(limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),

    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
])

def Train(config):
    gpu_manage(config)
    print('===> Loading datasets')
    dataset = TrainDataset(config, trfm)
    print('dataset:', len(dataset))
    train_size = int((1 - config.validation_size) * len(dataset))
    validation_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, validation_size])
    print('train dataset:', len(train_ds))
    print('validation dataset:', len(val_ds))
    training_data_loader = DataLoader(dataset=train_ds, num_workers=config.threads, batch_size=config.batchsize,
                                      shuffle=True)
    validation_data_loader = DataLoader(dataset=val_ds, num_workers=config.threads,
                                        batch_size=config.validation_batchsize, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = TMnet(num_res=config.nums_res)
    model = TMnet()
    criterionMSE = nn.MSELoss()
    l1loss = nn.L1Loss()
    if config.cuda:
        model = model.cuda()
        criterionMSE.cuda()
        l1loss.cuda()

    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=config.lr,
                                 betas=(0.9, 0.999),
                                 eps=1e-08,
                                 weight_decay=0.000003,
                                 amsgrad=False)
    logreport = lossReport(log_dir=config.out_dir)
    validationreport = TestReport(log_dir=config.out_dir)
    vald = TestReport('./result_t')
    print('===> begin')
    start_time = time.time()
    best_ssim = -1
    best_psnr = -1
    best_epoch = -1
    psnr = 0
    ssim = 0
    for epoch in range(config.epoch):
        epoch_start_time = time.time()
        adjust_learning_rate(optimizer, epoch, config.lr, 20)
        print_model(optimizer)
        losses = np.ones((len(training_data_loader)))
        for batch_idx, data in enumerate(training_data_loader):
            gt, Mask = data[0], data[1]
            gt = [g.cuda() for g in gt]
            Mask = [m.cuda() for m in Mask]
            cloud1 = gt[0] * Mask[0]
            cloud2 = gt[1] * Mask[1]
            cloud3 = gt[2] * Mask[2]
            cloud = [cloud1, cloud2, cloud3]
            torch.autograd.set_detect_anomaly(True)
            outputs = model(cloud) 
            '''
            l1 = preloss(gt[0], outputs[0], Mask[0])
            l2 = preloss(gt[1], outputs[1], Mask[1])
            l3 = preloss(gt[2], outputs[2], Mask[2])
            l4 = preloss(gt[0], outputs[3], Mask[0])
            l5 = preloss(gt[1], outputs[4], Mask[1])
            l6 = preloss(gt[2], outputs[5], Mask[2])
            loss_content = l1 + l2 + l3 + l4 + l5 + l6
            # loss_content = l1 + l2 + l3
            
            loss_fft = criterion_fft(outputs[0], gt[2]) + criterion_fft(outputs[1], gt[1]) + criterion_fft(outputs[2], gt[0])
            loss_content = 0.01 * loss_fft +  loss_content
            
            l1_ = l1loss(gt[0], outputs[0])
            l2_ = l1loss(gt[1], outputs[1])
            l3_ = l1loss(gt[2], outputs[2])
            l4_ = l1loss(gt[0], outputs[3])
            l5_ = l1loss(gt[1], outputs[4])
            l6_ = l1loss(gt[2], outputs[5])
            l11 = l1_ + l2_ + l3_ + l4_ + l5_ + l6_
            '''
            #l11 = l1_ + l2_ + l3_
            
            l1_ = l1loss(gt[2], outputs[3])
            l2_ = l1loss(gt[1], outputs[4])
            l3_ = l1loss(gt[0], outputs[5])
            l11_ = l1_ + l2_ + l3_
            loss_cr1 = crloss(outputs[3], gt[2], cloud3)
            loss_cr2 = crloss(outputs[4], gt[1], cloud2)
            loss_cr3 = crloss(outputs[5], gt[0], cloud1)
            loss_cr = loss_cr1 + loss_cr2 + loss_cr3
            loss_fft = criterion_fft(outputs[5], gt[0]) + criterion_fft(outputs[4], gt[1]) + criterion_fft(outputs[3], gt[2])
            loss = l11_ + 0.1 * loss_cr + loss_fft
            #loss = loss_content + l11
            # loss = loss_content * 10
            optimizer.zero_grad() #把梯度信息设置为0
            loss.backward() #反向传播
            optimizer.step()
            losses[batch_idx] = loss.item()
            print(
                "===> Epoch[{}]({}/{}): loss: {:.4f} ".format(
                    epoch, batch_idx, len(training_data_loader), loss.item()))
        log = {}
        log['epoch'] = epoch
        log['loss'] = np.average(losses)
        logreport(log)
        print('epoch', epoch, 'finished, use time', time.time() - epoch_start_time)
        # with torch.no_grad():
        #     print('train dataset:')
        #     log_validation, psnr, ssim = val(config, training_data_loader, model, criterionMSE, epoch)
        #     vald(log_validation)
        #     vald.save_lossgraph()
        with torch.no_grad():
            print('validation dataset:')
            log_validation, psnr, ssim = val(config, validation_data_loader, model, criterionMSE, epoch)
            validationreport(log_validation)
        if epoch % config.snapshot_interval == 0 or (epoch + 1) == config.epoch:
            checkpoint(config, epoch, model)
        if epoch % 1 == 0:
            if psnr >= best_psnr and ssim >= best_ssim:
                torch.save(model.state_dict(), os.path.join(config.out_dir + '/models', 'best.pth'))
                best_epoch = epoch
                best_psnr = psnr
                best_ssim = ssim
            print("best_epoch{}, best_psnr:{:.4f} dB, best_ssim:{:.4f}".format(best_epoch, best_psnr, best_ssim))
        logreport.save_lossgraph()
        validationreport.save_lossgraph()
    print('training time:', time.time() - start_time)


if __name__ == '__main__':
    with open('config.yml', 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = AttrMap(config)
    if config.is_train:
        Train(config)
