import numpy as np
from torch.autograd import Variable
import pytorch_ssim
import torch

ssim_loss = pytorch_ssim.SSIM()


def val(config, test_data_loader, model, criterionMSE, epoch):
    avg_mse = 0
    avg_psnr = 0
    avg_ssim = 0
    model.eval()
    for i, data in enumerate(test_data_loader):
        cloud = data[0]
        cloud = [c.cuda() for c in cloud]
        output = model(cloud)
        mse = criterionMSE(output[4], gt[1])
        # 1是指max_val
        psnr = 10 * np.log10(1 / mse.item())
        ssim = ssim_loss(output[4], gt[1])
        avg_mse += mse.item()
        avg_psnr += psnr
        avg_ssim += ssim
    avg_mse = avg_mse / len(test_data_loader)
    avg_psnr = avg_psnr / len(test_data_loader)
    avg_ssim = avg_ssim / len(test_data_loader)
    print("===> Avg. MSE: {:.4f}".format(np.sqrt(avg_mse)))
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))
    print("===> Avg. SSIM: {:.4f} ".format(avg_ssim))

    log_test = {}
    log_test['epoch'] = epoch
    log_test['mse'] = avg_mse
    log_test['psnr'] = avg_psnr
    # log_test['ssim'] = avg_ssim

    return log_test, avg_psnr, avg_ssim





# a = torch.ones((256,256))
# b = torch.ones((256,256))
# c = torch.ones((256,256))
# # for i in range(len(a)):
# #     for j in range(len(a[i])):
# #         if (i % 10 == 1):
# #             a[i][j] = 0
# #         if (i % 10 == 1 or j % 10 == 1):
# #             b[i][j] = 0
# #         if (i % 10 == 1 or i % 10 == 2 or j % 10 == 1):
# #             c[i][j] = 0
# for i in range(len(a)):
#     for j in range(len(a[i])):
#         if (i % 10 == 1 or i % 10 == 2 or i % 10 == 3 or j % 10 == 3 or i % 10 == 2 or j % 10 == 1):
#             a[i][j] = 0
#         if (i % 10 == 1 or i % 10 == 2 or i % 10 == 3 or j % 10 == 3 or i % 10 == 2 or j % 10 == 1 or i % 10 == 4):
#             b[i][j] = 0
#         if (i % 10 == 1 or i % 10 == 2 or i % 10 == 3 or j % 10 == 3 or i % 10 == 2 or j % 10 == 1 or i % 10 == 4 or j % 10 == 4):
#             c[i][j] = 0
# mask_s = list()
# mask_s.append(a)
# mask_s.append(b)
# mask_s.append(c)

# def val(config, test_data_loader, model, criterionMSE, epoch):
#     avg_mse = 0
#     avg_psnr = 0
#     avg_ssim = 0
#     model.eval()
#     for i, data in enumerate(test_data_loader):
#         gt, Mask = data[0], data[1]
#         gt = [g.cuda() for g in gt]
#         Mask = [m.cuda() for m in Mask]
#         mask_s = [m.cuda() for m in mask_s]
#         # cloud1 = gt[0] * Mask[0]
#         # cloud2 = gt[1] * Mask[1]
#         # cloud3 = gt[2] * Mask[2]
#         # cloud = [cloud1, cloud2, cloud3]
#         cloud = list()
#         cloud.append(mask_s[0] * gt[0])
#         cloud.append(mask_s[1] * gt[1])
#         cloud.append(mask_s[2] * gt[2])
#         output = model(cloud)
#         mse = criterionMSE(output[1], gt[1])
#         # 1是指max_val
#         psnr = 10 * np.log10(1 / mse.item())
#         ssim = ssim_loss(output[1], gt[1])
#         avg_mse += mse.item()
#         avg_psnr += psnr
#         avg_ssim += ssim
#     avg_mse = avg_mse / len(test_data_loader)
#     avg_psnr = avg_psnr / len(test_data_loader)
#     avg_ssim = avg_ssim / len(test_data_loader)
#     print("===> Avg. MSE: {:.4f}".format(np.sqrt(avg_mse)))
#     print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))
#     print("===> Avg. SSIM: {:.4f} ".format(avg_ssim))

#     log_test = {}
#     log_test['epoch'] = epoch
#     log_test['mse'] = avg_mse
#     log_test['psnr'] = avg_psnr
#     # log_test['ssim'] = avg_ssim

#     return log_test, avg_psnr, avg_ssim
