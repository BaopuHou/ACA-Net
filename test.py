import os
import time
import matplotlib
import numpy as np
import torch
import torch.backends.cudnn
import torch.utils.data
import torch.nn as nn
import yaml
from attrdict import AttrMap
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as SSIM
from torch.utils.data import DataLoader
from Evaluation_Index import Evaluation_index
from modules.ACA import TMnet
from utils.datasets import tDataset, rDataset
from utils.lossReport import TestReport
from utils.utils import save_bmp, mkdir#, truncated_linear_stretch

matplotlib.use('Agg')


def test(config, test_data_loader, model_path, outpath, log_name='test', level=1, is_truncat=False):
    avg_rmse = 0
    avg_psnr = 0
    avg_ssim = 0
    avg_cc = 0
    index = 3
    device = torch.device("cuda:0" if (config.cuda) else "cpu")
    #print(torch.__version__)
    # model = TSRnet(num_res=config.nums_res)
    model = TMnet()#num_res=config.nums_res)
    model.to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device), False)
    # print(model)
    model.eval()
    log_test = {}
    validationreport = TestReport(log_dir=config.test_out_dir + '/Evaluation/', log_name=log_name)
    for i, data in enumerate(test_data_loader):
        gt, Mask, name = data[0], data[1], data[2]
        if config.cuda:
            gt = [g.cuda() for g in gt]
            Mask = [m.cuda() for m in Mask]
        with torch.no_grad():
            cloud1 = gt[0] * Mask[0]
            cloud2 = gt[1] * Mask[1]
            cloud3 = gt[2] * Mask[2]
            cloud = [cloud1, cloud2, cloud3]
            output = model(cloud)
        if level == 1:
            index = 5
        if level == 2:
            index = 4
        if level == 3:
            index = 3
        img1 = output[index].cpu().detach().numpy()[0, 0, :, :]

        img2 = gt[level - 1].cpu().detach().numpy()[0, 0, :, :]
        # print(img1, img2)
        if config.is_save_pre:
            m = Mask[level - 1].cpu().detach().numpy()[0, 0, :, :]
            img1 = img1 * (1 - m) + img2 * m
        cloud_numbers = np.where(Mask[level - 1].cpu().detach().numpy()[0, 0, :, :] == 0)[0].size
        w, h = Mask[level - 1].cpu().detach().numpy()[0, 0, :, :].shape
        precent_cloud = (cloud_numbers / (w * h)) * 100
        # print(precent_cloud)
        cl = cloud[level - 1].cpu().detach().numpy()[0, 0, :, :]
        ssim = SSIM(img2, img1,data_range=1)
        psnr = peak_signal_noise_ratio(img2, img1, data_range=1)
        # print(img1)
        cc, rmse = Evaluation_index(img1, img2)
        print(name[0], ": ssim:{:.4f}, psnr:{:.4f} dB, cc:{:.4f}, rmse:{:.7f}".format(ssim, psnr, cc, rmse))
        if not os.path.exists(config.test_out_dir + 'ACA-Net' + '/' + outpath):
            mkdir(config.test_out_dir + 'ACA-Net' + '/' + outpath + '/' + 'result')
            mkdir(config.test_out_dir + 'ACA-Net' + '/' + outpath + '/' + 'gt')
            mkdir(config.test_out_dir + 'ACA-Net' + '/' + outpath + '/' + 'cloud')
        if is_truncat:
            img1 = img1 * 255
            img2 = img2 * 255
            cl = cl * 255
            img1 = truncated_linear_stretch(img1, 1)
            img2 = truncated_linear_stretch(img2, 1)
            cl = truncated_linear_stretch(cl, 1)
            save_bmp(img1, config.test_out_dir + 'ACA-Net' + '/' + outpath + '/' + 'result' + '/result_' + name[0])
            save_bmp(img2, config.test_out_dir + 'ACA-Net' + '/' + outpath + '/' + 'gt' + '/gt_' + name[0])
            save_bmp(cl, config.test_out_dir + 'ACA-Net' + '/' + outpath + '/' + 'cloud' + '/cloud_' + name[0])

        else:
            # img1 = img1 * 255
            # img2 = img2 * 255
            # cl = cl * 255
            # print(img1)
            save_bmp(img1, config.test_out_dir + 'ACA-Net' + '/' + outpath + '/' + 'result' + '/result_' + name[0])
            save_bmp(img2, config.test_out_dir + 'ACA-Net' + '/' + outpath + '/' + 'gt' + '/gt_' + name[0])
            save_bmp(cl, config.test_out_dir + 'ACA-Net' + '/' + outpath + '/' + 'cloud' + '/cloud_' + name[0])

        # plt.subplot(1, 3, 1), plt.imshow(img1, 'gray'), plt.title('remove_cloud')
        # plt.axis('off')
        # plt.subplot(1, 3, 2), plt.imshow(img2, 'gray'), plt.title('gt')
        # plt.axis('off')
        # plt.subplot(1, 3, 3), plt.imshow(cl, 'gray'), plt.title('cloud')
        # plt.axis('off')
        # plt.show()
        # plt.close()

        avg_psnr += psnr
        avg_ssim += ssim
        avg_cc += cc
        avg_rmse += rmse

        log_test['img'] = name[0]
        log_test['rmse'] = rmse
        log_test['psnr'] = psnr
        log_test['ssim'] = ssim
        log_test['cc'] = cc
        log_test['cloud'] = precent_cloud
        validationreport(log_test)
        log_test = {}

    avg_cc = avg_cc / len(test_data_loader)
    avg_rmse = avg_rmse / len(test_data_loader)
    avg_ssim = avg_ssim / len(test_data_loader)
    avg_psnr = avg_psnr / len(test_data_loader)
    #保存平均指标到新的文件中
    with open(config.test_out_dir + 'ACA-Net' + '/' + outpath + '/' + log_name + '.txt', 'a') as f:
        f.write("average accuracy: psnr:{:.4f} dB, ssim:{:.4f}, cc:{:.4f}, rmse:{}\n".format(avg_psnr, avg_ssim, avg_cc, avg_rmse))
    
    print("average accuracy: psnr:{:.4f} dB, ssim:{:.4f}, cc:{:.4f}, rmse:{}".
          format(avg_psnr, avg_ssim, avg_cc, avg_rmse))



def test_real(config, test_data_loader, model_path, outpath, log_name='test', level=1, is_truncat=False):
    device = torch.device("cuda:0" if (config.cuda) else "cpu")
    # model = TSRnet(num_res=config.nums_res)
    # model = TMnet(num_res=config.nums_res)
    model = ACA-Net()
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    log_test = {}
    validationreport = TestReport(log_dir=config.test_out_dir + '/Evaluation/', log_name=log_name)
    for i, data in enumerate(test_data_loader):
        gt, Mask, name, path1, path2, path3 = data[0], data[1], data[2], data[3], data[4], data[5]
        if config.cuda:
            gt = [g.cuda() for g in gt]
            Mask = [m.cuda() for m in Mask]
        with torch.no_grad():
            cloud1 = gt[0] * Mask[0]
            cloud2 = gt[1] * Mask[1]
            cloud3 = gt[2] * Mask[2]
            cloud = [cloud1, cloud2, cloud3]
            output = model(cloud)
        img1 = output[5].cpu().detach().numpy()[0, 0, :, :]
        img11 = output[4].cpu().detach().numpy()[0, 0, :, :]
        img111 = output[3].cpu().detach().numpy()[0, 0, :, :]
        img2 = gt[0].cpu().detach().numpy()[0, 0, :, :]
        img22 = gt[1].cpu().detach().numpy()[0, 0, :, :]
        img222 = gt[2].cpu().detach().numpy()[0, 0, :, :]

        if config.is_save_pre:
            m = Mask[0].cpu().detach().numpy()[0, 0, :, :]
            m1 = Mask[1].cpu().detach().numpy()[0, 0, :, :]
            m11 = Mask[2].cpu().detach().numpy()[0, 0, :, :]
            img1 = img1 * (1 - m) + img2 * m
            img11 = img11 * (1 - m1) + img22 * m1
            img111 = img111 * (1 - m11) + img222 * m11

        cloud_numbers1 = np.where(Mask[0].cpu().detach().numpy()[0, 0, :, :] == 0)[0].size
        cloud_numbers2 = np.where(Mask[1].cpu().detach().numpy()[0, 0, :, :] == 0)[0].size
        cloud_numbers3 = np.where(Mask[2].cpu().detach().numpy()[0, 0, :, :] == 0)[0].size
        w, h = Mask[0].cpu().detach().numpy()[0, 0, :, :].shape
        precent_cloud1 = (cloud_numbers1 / (w * h)) * 100
        precent_cloud2 = (cloud_numbers2 / (w * h)) * 100
        precent_cloud3 = (cloud_numbers3 / (w * h)) * 100

        cl = cloud1.cpu().detach().numpy()[0, 0, :, :]
        cl1 = cloud2.cpu().detach().numpy()[0, 0, :, :]
        cl11 = cloud3.cpu().detach().numpy()[0, 0, :, :]
        print(name[0])

        if not os.path.exists(config.test_out_dir + 'ACA-Net' + '/' + path1[0]):
            mkdir(config.test_out_dir + 'ACA-Net' + '/' + path1[0] + '/' + 'result')
            mkdir(config.test_out_dir + 'ACA-Net' + '/' + path1[0] + '/' + 'gt')
            mkdir(config.test_out_dir + 'ACA-Net' + '/' + path1[0] + '/' + 'cloud')
            mkdir(config.test_out_dir + 'ACA-Net' + '/' + path2[0] + '/' + 'result')
            mkdir(config.test_out_dir + 'ACA-Net' + '/' + path2[0] + '/' + 'gt')
            mkdir(config.test_out_dir + 'ACA-Net' + '/' + path2[0] + '/' + 'cloud')
            mkdir(config.test_out_dir + 'ACA-Net' + '/' + path3[0] + '/' + 'result')
            mkdir(config.test_out_dir + 'ACA-Net' + '/' + path3[0] + '/' + 'gt')
            mkdir(config.test_out_dir + 'ACA-Net' + '/' + path3[0] + '/' + 'cloud')

        if is_truncat:
            img1 = truncated_linear_stretch(img1 * 255, 1)
            img2 = truncated_linear_stretch(img2 * 255, 1)
            cl = truncated_linear_stretch(cl * 255, 1)
            save_bmp(img1, config.test_out_dir + 'ACA-Net' + '/' + path1[0] + '/' + 'result' + '/result_' + name[0])
            save_bmp(img2, config.test_out_dir + 'ACA-Net' + '/' + path1[0] + '/' + 'gt' + '/gt_' + name[0])
            save_bmp(cl, config.test_out_dir + 'ACA-Net' + '/' + path1[0] + '/' + 'cloud' + '/cloud_' + name[0])

        else:
            save_bmp(img1 , config.test_out_dir + 'ACA-Net' + '/' + path1[0] + '/' + 'result' + '/result_' + name[0])
            save_bmp(img2, config.test_out_dir + 'ACA-Net' + '/' + path1[0] + '/' + 'gt' + '/gt_' + name[0])
            save_bmp(cl, config.test_out_dir + 'ACA-Net' + '/' + path1[0] + '/' + 'cloud' + '/cloud_' + name[0])
            save_bmp(img11 , config.test_out_dir + 'ACA-Net' + '/' + path2[0] + '/' + 'result' + '/result_' + name[0])
            save_bmp(img22 , config.test_out_dir + 'ACA-Net' + '/' + path2[0] + '/' + 'gt' + '/gt_' + name[0])
            save_bmp(cl1 , config.test_out_dir + 'ACA-Net' + '/' + path2[0] + '/' + 'cloud' + '/cloud_' + name[0])
            save_bmp(img111, config.test_out_dir + 'ACA-Net' + '/' + path3[0] + '/' + 'result' + '/result_' + name[0])
            save_bmp(img222, config.test_out_dir + 'ACA-Net' + '/' + path3[0] + '/' + 'gt' + '/gt_' + name[0])
            save_bmp(cl11, config.test_out_dir + 'ACA-Net' + '/' + path3[0] + '/' + 'cloud' + '/cloud_' + name[0])

        # log_test['img'] = name[0]
        log_test[path1[0]] = precent_cloud1
        log_test[path2[0]] = precent_cloud2
        log_test[path3[0]] = precent_cloud3
        validationreport(log_test)
        log_test = {}


if __name__ == '__main__':
    with open('config.yml', 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = AttrMap(config)

    #modelpath = os.path.join(config.test_out_dir, 'models', 'best.pth')
    modelpath = r'./result/modules/best.pth'  
    # modelpath = r'./result/models/best.pth'
    is_real = False
    # is_real = True
    start_time = time.time()
    if is_real:
        outpathname = 'bj-TM-real-0708-2'
        testset = rDataset(config, is_real=is_real)
        test_data_loader = DataLoader(dataset=testset, num_workers=1,
                                      batch_size=1, shuffle=False)
        test_real(config, test_data_loader, modelpath, outpathname,
                  log_name='bj-TM-real-0708-2', level=2, is_truncat=False)
        print('avg test time:', (time.time() - start_time) / len(test_data_loader))
    else:
        outpathname = 'KPL-1'
        # outpathname = 'TC-1'
        #outpathname = os.path.join(config.test_out_dir, 't')
        testset = tDataset(config, is_real=False)
        test_data_loader = DataLoader(dataset=testset, num_workers=0,
                                      batch_size=1, shuffle=False)
        test(config, test_data_loader, modelpath, outpathname,
             log_name='l1', level=1, is_truncat=False)
        outpathname = 'KPL-2'
        # outpathname = 'TC-2'
        test(config, test_data_loader, modelpath, outpathname,
             log_name='l2', level=2, is_truncat=False)
        outpathname = 'KPL-3'
        # outpathname = 'TC-3'
        test(config, test_data_loader, modelpath, outpathname,
             log_name='l3', level=3, is_truncat=False)
        print('avg test time:', (time.time() - start_time) / len(test_data_loader) / 3)
