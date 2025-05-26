import cv2
import random
import numpy as np
import torch
import os
from torch.utils import data
from torchvision import transforms as T
from PIL import Image
import tifffile as tiff
import albumentations as A

class MyDataset(data.Dataset):

    def __init__(self, data_root, mode='train'):
        super().__init__()
        self.data_root = data_root
        self.mode = mode
        self.transform = A.Compose([
            A.Rotate(limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),

            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ])
        if  mode == 'train':
            train_list_file = os.path.join(data_root,'train','gt', 'train_list.txt')
            # 如果数据集尚未分割，则进行训练集和测试集的分割
            if not os.path.exists(train_list_file) or os.path.getsize(train_list_file) == 0:
                files = os.listdir(os.path.join(data_root,'train','gt', '1'))
                random.shuffle(files)
                n_train = int(0.95 * len(files))
                train_list = files[:n_train]
                val_list = files[n_train:]
                np.savetxt(os.path.join(data_root,'train','gt', 'train_list.txt' ), np.array(train_list), fmt='%s')
                np.savetxt(os.path.join(data_root,'train','gt', 'val_list.txt' ), np.array(val_list), fmt='%s')
            self.imlist = np.loadtxt(train_list_file, str)
        elif mode == 'val':
            val_list_file = os.path.join(data_root,'train','gt', 'val_list.txt')
            self.imlist = np.loadtxt(val_list_file, str)
        elif mode == 'test':
            test_list_file = os.path.join(data_root, 'test', 'test_list.txt')
            if not os.path.exists(test_list_file) or os.path.getsize(test_list_file) == 0:
                files = os.listdir(os.path.join(data_root, 'test', '20211121'))
                files.sort(key=lambda x: int(x[:-4]))
                np.savetxt(os.path.join(data_root, 'test', 'test_list.txt'), np.array(files), fmt='%s')
            self.imlist = np.loadtxt(test_list_file, str)
        

    def __getitem__(self, index):
        if self.mode == 'train':
            cloud = cv2.imread(os.path.join(self.data_root,'train','gt', '1', str(self.imlist[index])),
                               cv2.IMREAD_GRAYSCALE).astype(np.float32)
            x1 = cv2.imread(os.path.join(self.data_root,'train','gt', '2', str(self.imlist[index])),
                            cv2.IMREAD_GRAYSCALE).astype(np.float32)
            x2 = cv2.imread(os.path.join(self.data_root,'train','gt', '3', str(self.imlist[index])),
                            cv2.IMREAD_GRAYSCALE).astype(np.float32)

            m = cv2.imread(os.path.join(self.data_root,'train','mask', '1',
                                        (str(self.imlist[index])[:-4] + '.bmp')), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            m1 = cv2.imread(os.path.join(self.data_root,'train','mask', '2',
                                         (str(self.imlist[index])[:-4] + '.bmp')), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            m2 = cv2.imread(os.path.join(self.data_root,'train','mask', '3',
                                         (str(self.imlist[index])[:-4] + '.bmp')), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        elif self.mode == 'val':
            cloud = cv2.imread(os.path.join(self.data_root,'train','gt', '1', str(self.imlist[index])),
                               cv2.IMREAD_GRAYSCALE).astype(np.float32)
            x1 = cv2.imread(os.path.join(self.data_root,'train','gt', '2', str(self.imlist[index])),
                            cv2.IMREAD_GRAYSCALE).astype(np.float32)
            x2 = cv2.imread(os.path.join(self.data_root,'train','gt', '3', str(self.imlist[index])),
                            cv2.IMREAD_GRAYSCALE).astype(np.float32)

            m = cv2.imread(os.path.join(self.data_root,'train','mask', '1',
                                        (str(self.imlist[index])[:-4] + '.bmp')), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            m1 = cv2.imread(os.path.join(self.data_root,'train','mask', '2',
                                         (str(self.imlist[index])[:-4] + '.bmp')), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            m2 = cv2.imread(os.path.join(self.data_root,'train','mask', '3',
                                         (str(self.imlist[index])[:-4] + '.bmp')), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        elif self.mode == 'test':
            cloud = cv2.imread(os.path.join(self.data_root,'test', '20211121', str(self.imlist[index])),
                               cv2.IMREAD_GRAYSCALE).astype(np.float32)
            x1 = cv2.imread(os.path.join(self.data_root,'test', '20211203',  str(self.imlist[index])),
                            cv2.IMREAD_GRAYSCALE).astype(np.float32)
            x2 = cv2.imread(os.path.join(self.data_root,'test', '20211211', str(self.imlist[index])),
                            cv2.IMREAD_GRAYSCALE).astype(np.float32)

            m = cv2.imread(os.path.join(self.data_root,'test', '1',
                                        (str(self.imlist[index])[:-4] + '.bmp')), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            m1 = cv2.imread(os.path.join(self.data_root,'test', '2',
                                         (str(self.imlist[index])[:-4] + '.bmp')), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            m2 = cv2.imread(os.path.join(self.data_root,'test', '3',
                                         (str(self.imlist[index])[:-4] + '.bmp')), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        m = 1 - (m / 255)
        m1 = (1 - (m1 / 255))
        m2 = (1 - (m2 / 255))

        # 归一化
        temp = np.dstack((cloud, x1, x2)) / 255
        m = np.dstack((m, m1, m2))
        if self.mode == 'test':
            gt1 = torch.from_numpy(np.transpose(temp, (2, 0, 1))[0:1, :, :])
            gt2 = torch.from_numpy(np.transpose(temp, (2, 0, 1))[1:2, :, :])
            gt3 = torch.from_numpy(np.transpose(temp, (2, 0, 1))[2:3, :, :])
            gt = [gt1, gt2, gt3]
            mask1 = torch.from_numpy(np.transpose(m, (2, 0, 1))[0:1, :, :])
            mask2 = torch.from_numpy(np.transpose(m, (2, 0, 1))[1:2, :, :])
            mask3 = torch.from_numpy(np.transpose(m, (2, 0, 1))[2:3, :, :])
            m = [mask1, mask2, mask3]
        else:
            augments = self.transform(image=temp, mask=m)
            gt1 = torch.from_numpy(np.transpose(augments['image'], (2, 0, 1))[0:1, :, :])
            gt2 = torch.from_numpy(np.transpose(augments['image'], (2, 0, 1))[1:2, :, :])
            gt3 = torch.from_numpy(np.transpose(augments['image'], (2, 0, 1))[2:3, :, :])
            gt = [gt1, gt2, gt3]
            mask1 = torch.from_numpy(np.transpose(augments['mask'], (2, 0, 1))[0:1, :, :])
            mask2 = torch.from_numpy(np.transpose(augments['mask'], (2, 0, 1))[1:2, :, :])
            mask3 = torch.from_numpy(np.transpose(augments['mask'], (2, 0, 1))[2:3, :, :])
            m = [mask1, mask2, mask3]
        cloud1 = gt[0] * m[0]
        cloud2 = gt[1] * m[1]
        cloud3 = gt[2] * m[2]
        #把cloud1变为为三通道
        # cloud1 = torch.cat([cloud1, cloud1, cloud1], dim=0)
        # cloud2 = torch.cat([cloud2, cloud2, cloud2], dim=0)
        # cloud3 = torch.cat([cloud3, cloud3, cloud3], dim=0)
        # gt[0] = torch.cat([gt[0], gt[0], gt[0]], dim=0)


        
        return [cloud1, cloud2, cloud3],gt, str(self.imlist[index])
        # ret = {}
        # ret['gt_image'] = gt[0]
        # ret['cond_image'] = torch.cat([cloud1, cloud2, cloud3])
        # ret['path'] = str(self.imlist[index])
        # return ret

    def __len__(self):
        return len(self.imlist)

class TrainDataset(data.Dataset):

    def __init__(self, config, transform):
        super().__init__()
        self.config = config
        self.transform = transform
        train_list_file = os.path.join(config.datasets_dir, config.train_list)
        # 如果数据集尚未分割，则进行训练集和测试集的分割
        if not os.path.exists(train_list_file) or os.path.getsize(train_list_file) == 0:
            files = os.listdir(os.path.join(config.datasets_dir, config.years[0]))
            random.shuffle(files)
            n_train = int(config.train_size * len(files))
            train_list = files[:n_train]
            test_list = files[n_train:]
            np.savetxt(os.path.join(config.datasets_dir, config.train_list), np.array(train_list), fmt='%s')
            np.savetxt(os.path.join(config.datasets_dir, config.test_list), np.array(test_list), fmt='%s')
        self.imlist = np.loadtxt(train_list_file, str)

    def __getitem__(self, index):
        cloud = cv2.imread(os.path.join(self.config.datasets_dir, self.config.years[0], str(self.imlist[index])),
                           cv2.IMREAD_GRAYSCALE).astype(np.float32)
        x1 = cv2.imread(os.path.join(self.config.datasets_dir, self.config.years[1], str(self.imlist[index])),
                        cv2.IMREAD_GRAYSCALE).astype(np.float32)
        x2 = cv2.imread(os.path.join(self.config.datasets_dir, self.config.years[2], str(self.imlist[index])),
                        cv2.IMREAD_GRAYSCALE).astype(np.float32)
        #将图片大小改为128*128
        # cloud = cv2.resize(cloud, (128, 128), interpolation=cv2.INTER_CUBIC)
        # x1 = cv2.resize(x1, (128, 128), interpolation=cv2.INTER_CUBIC)
        # x2 = cv2.resize(x2, (128, 128), interpolation=cv2.INTER_CUBIC)

        m = cv2.imread(os.path.join(self.config.mask_dir, self.config.mask_dir_name[0],
                                    (str(self.imlist[index])[:-4] + '.bmp')), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        m1 = cv2.imread(os.path.join(self.config.mask_dir, self.config.mask_dir_name[1],
                                     (str(self.imlist[index])[:-4] + '.bmp')), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        m2 = cv2.imread(os.path.join(self.config.mask_dir, self.config.mask_dir_name[2],
                                     (str(self.imlist[index])[:-4] + '.bmp')), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        #将mask大小改为128*128
        # m = cv2.resize(m, (128, 128), interpolation=cv2.INTER_CUBIC)
        # m1 = cv2.resize(m1, (128, 128), interpolation=cv2.INTER_CUBIC)
        # m2 = cv2.resize(m2, (128, 128), interpolation=cv2.INTER_CUBIC)

        m = 1 - (m / 255)
        m1 = (1 - (m1 / 255))
        m2 = (1 - (m2 / 255))

        # 归一化
        temp = np.dstack((cloud, x1, x2)) / 255
        m = np.dstack((m, m1, m2))
        augments = self.transform(image=temp, mask=m)
        gt1 = torch.from_numpy(np.transpose(augments['image'], (2, 0, 1))[0:1, :, :])
        gt2 = torch.from_numpy(np.transpose(augments['image'], (2, 0, 1))[1:2, :, :])
        gt3 = torch.from_numpy(np.transpose(augments['image'], (2, 0, 1))[2:3, :, :])
        gt = [gt1, gt2, gt3]
        mask1 = torch.from_numpy(np.transpose(augments['mask'], (2, 0, 1))[0:1, :, :])
        mask2 = torch.from_numpy(np.transpose(augments['mask'], (2, 0, 1))[1:2, :, :])
        mask3 = torch.from_numpy(np.transpose(augments['mask'], (2, 0, 1))[2:3, :, :])
        m = [mask1, mask2, mask3]
        return gt, m

    def __len__(self):
        return len(self.imlist)


class tDataset(data.Dataset):

    def __init__(self, config, is_real=False):
        super().__init__()
        self.config = config
        self.is_real = is_real
        test_list_file = os.path.join(config.test_dir, config.test_list)
        if not os.path.exists(test_list_file) or os.path.getsize(test_list_file) == 0:
            files = os.listdir(os.path.join(config.test_dir, config.cloud[0]))
            files.sort(key=lambda x: int(x[:-4]))
            # np.savetxt(os.path.join(config.test_dir, config.train_list), np.array(files), fmt='%s')
            np.savetxt(os.path.join(config.test_dir, config.test_list), np.array(files), fmt='%s')
        # else:
        #     os.remove(test_list_file)
        #     files = os.listdir(os.path.join(config.test_dir, config.cloud[0]))
        #     files.sort(key=lambda x: int(x[:-4]))
        #     # np.savetxt(os.path.join(config.test_dir, config.train_list), np.array(files), fmt='%s')
        #     np.savetxt(os.path.join(config.test_dir, config.test_list), np.array(files), fmt='%s')
        self.imlist = np.loadtxt(test_list_file, str)

    def __getitem__(self, index):
        # [index]
        print(os.path.join(self.config.test_dir, self.config.cloud[0], str(self.imlist[index])))
        cloud = cv2.imread(os.path.join(self.config.test_dir, self.config.cloud[0], str(self.imlist[index])),
                           cv2.IMREAD_GRAYSCALE).astype(np.float32)
        x1 = cv2.imread(os.path.join(self.config.test_dir, self.config.cloud[1], str(self.imlist[index])),
                        cv2.IMREAD_GRAYSCALE).astype(np.float32)
        x2 = cv2.imread(os.path.join(self.config.test_dir, self.config.cloud[2], str(self.imlist[index])),
                        cv2.IMREAD_GRAYSCALE).astype(np.float32)
        #将图片大小改为128*128
        # cloud = cv2.resize(cloud, (128, 128), interpolation=cv2.INTER_CUBIC)
        # x1 = cv2.resize(x1, (128, 128), interpolation=cv2.INTER_CUBIC)
        # x2 = cv2.resize(x2, (128, 128), interpolation=cv2.INTER_CUBIC)
        if self.is_real:
            end = '.tif'
        else:
            end = '.bmp'
        # print(os.path.join(self.config.tmask_dir, self.config.mask_dir_name[1],
        #                             (str(self.imlist[index])[:-4] + end)))
        m = cv2.imread(os.path.join(self.config.tmask_dir, self.config.mask_dir_name[0],
                                    (str(self.imlist[index])[:-4] + end)), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        m1 = cv2.imread(os.path.join(self.config.tmask_dir, self.config.mask_dir_name[1],
                                     (str(self.imlist[index])[:-4] + end)), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        m2 = cv2.imread(os.path.join(self.config.tmask_dir, self.config.mask_dir_name[2],
                                     (str(self.imlist[index])[:-4] + end)), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        #将mask大小改为128*128
        # m = cv2.resize(m, (128, 128), interpolation=cv2.INTER_CUBIC)
        # m1 = cv2.resize(m1, (128, 128), interpolation=cv2.INTER_CUBIC)
        # m2 = cv2.resize(m2, (128, 128), interpolation=cv2.INTER_CUBIC)
        m = 1 - (m / 255)
        m1 = 1 - (m1 / 255)
        m2 = 1 - (m2 / 255)

        # 归一化
        temp = np.dstack((cloud, x1, x2)) / 255
        m = np.dstack((m, m1, m2))
        # augments = self.transform(image=temp, mask=m)
        gt1 = torch.from_numpy(np.transpose(temp, (2, 0, 1))[0:1, :, :])
        gt2 = torch.from_numpy(np.transpose(temp, (2, 0, 1))[1:2, :, :])
        gt3 = torch.from_numpy(np.transpose(temp, (2, 0, 1))[2:3, :, :])
        gt = [gt1, gt2, gt3]
        mask1 = torch.from_numpy(np.transpose(m, (2, 0, 1))[0:1, :, :])
        mask2 = torch.from_numpy(np.transpose(m, (2, 0, 1))[1:2, :, :])
        mask3 = torch.from_numpy(np.transpose(m, (2, 0, 1))[2:3, :, :])
        m = [mask1, mask2, mask3]
        name_ = str(self.imlist[index])[:-4] + '.tif'
        if self.is_real:
            name_ = str(self.imlist[index])[:-4] + '.tif'
        return gt, m, name_

    def __len__(self):
        return len(self.imlist)

class realDataset(data.Dataset):

    def __init__(self, is_real=True):
        super().__init__()
        self.is_real = is_real
        self.target_path_1 = "D:\CODE\JJH\\real\\real\\1"
        self.target_path_2 = "D:\CODE\JJH\\real\\real\\2"
        self.target_path_3 = "D:\CODE\JJH\\real\\real\\3"
        self.target_path_4 = "D:\CODE\JJH\\real\\real\\4"
        self.target_path_5 = "D:\CODE\JJH\\real\\real\\mask_1"
        self.target_path_6 = "D:\CODE\JJH\\real\\real\\mask_2"
        self.target_path_7 = "D:\CODE\JJH\\real\\real\\mask_3"
        self.target_path_8 = "D:\CODE\JJH\\real\\real\\mask_4"
        test_path = "D:\CODE\JJH\\real\\real\\test_list.txt"
        files = os.listdir(self.target_path_1)
            # np.savetxt(os.path.join(config.test_dir, config.train_list), np.array(files), fmt='%s')
        tif_files = [file for file in files if file.endswith('.tif')]
        np.savetxt(test_path, np.array(tif_files), fmt='%s')
        #读取test_path中以.tif结尾的文件
        self.imlist = np.loadtxt(test_path, str)

    def __getitem__(self, index):
        gt1 = cv2.imread(os.path.join(self.target_path_1, str(self.imlist[index])), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        gt2 = cv2.imread(os.path.join(self.target_path_2, str(self.imlist[index])), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        gt3 = cv2.imread(os.path.join(self.target_path_3, str(self.imlist[index])), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        gt4 = cv2.imread(os.path.join(self.target_path_4, str(self.imlist[index])), cv2.IMREAD_GRAYSCALE).astype(np.float32)

        mask1 = cv2.imread(os.path.join(self.target_path_5, str((self.imlist[index])[:-4] + '.bmp')), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        mask2 = cv2.imread(os.path.join(self.target_path_6, str((self.imlist[index])[:-4] + '.bmp')), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        mask3 = cv2.imread(os.path.join(self.target_path_7, str((self.imlist[index])[:-4] + '.bmp')), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        mask4 = cv2.imread(os.path.join(self.target_path_8, str((self.imlist[index])[:-4] + '.bmp')), cv2.IMREAD_GRAYSCALE).astype(np.float32)

        mask1 = 1 - (mask1 / 255)
        mask2 = 1 - (mask2 / 255)
        mask3 = 1 - (mask3 / 255)
        mask4 = 1 - (mask4 / 255)

        # 归一化
        temp = np.dstack((gt1, gt2, gt3, gt4)) / 255
        m = np.dstack((mask1, mask2, mask3, mask4))
        gt1 = torch.from_numpy(np.transpose(temp, (2, 0, 1))[0:1, :, :])
        gt2 = torch.from_numpy(np.transpose(temp, (2, 0, 1))[1:2, :, :])
        gt3 = torch.from_numpy(np.transpose(temp, (2, 0, 1))[2:3, :, :])
        gt4 = torch.from_numpy(np.transpose(temp, (2, 0, 1))[3:4, :, :])
        gt = [gt1, gt2, gt3]
        mask1 = torch.from_numpy(np.transpose(m, (2, 0, 1))[0:1, :, :])
        mask2 = torch.from_numpy(np.transpose(m, (2, 0, 1))[1:2, :, :])
        mask3 = torch.from_numpy(np.transpose(m, (2, 0, 1))[2:3, :, :])
        mask4 = torch.from_numpy(np.transpose(m, (2, 0, 1))[3:4, :, :])
        m = [mask1, mask2, mask3]
        return gt, m, str(self.imlist[index])
    def __len__(self):
        return len(self.imlist)

class rDataset(data.Dataset):
    def __init__(self, config, is_real=False):
        super().__init__()
        self.config = config
        self.is_real = is_real
        test_list_file = os.path.join(config.test_dir, config.test_list)
        if not os.path.exists(test_list_file) or os.path.getsize(test_list_file) == 0:
            files = os.listdir(os.path.join(config.test_dir, config.cloud[0]))
            # np.savetxt(os.path.join(config.test_dir, config.train_list), np.array(files), fmt='%s')
            np.savetxt(os.path.join(config.test_dir, config.test_list), np.array(files), fmt='%s')
        else:
            os.remove(test_list_file)
            files = os.listdir(os.path.join(config.test_dir, config.cloud[0]))
            # np.savetxt(os.path.join(config.test_dir, config.train_list), np.array(files), fmt='%s')
            np.savetxt(os.path.join(config.test_dir, config.test_list), np.array(files), fmt='%s')
        self.imlist = np.loadtxt(test_list_file, str)

    def __getitem__(self, index):
        # [index]
        cloud = cv2.imread(os.path.join(self.config.test_dir, self.config.cloud[0], str(self.imlist[index])),
                           cv2.IMREAD_GRAYSCALE).astype(np.float32)
        x1 = cv2.imread(os.path.join(self.config.test_dir, self.config.cloud[1], str(self.imlist[index])),
                        cv2.IMREAD_GRAYSCALE).astype(np.float32)
        x2 = cv2.imread(os.path.join(self.config.test_dir, self.config.cloud[2], str(self.imlist[index])),
                        cv2.IMREAD_GRAYSCALE).astype(np.float32)

        m = cv2.imread(os.path.join(self.config.tmask_dir, self.config.mask_dir_name[0],
                                    (str(self.imlist[index])[:-4] + '.tif')), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        m1 = cv2.imread(os.path.join(self.config.tmask_dir, self.config.mask_dir_name[1],
                                     (str(self.imlist[index])[:-4] + '.tif')), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        m2 = cv2.imread(os.path.join(self.config.tmask_dir, self.config.mask_dir_name[2],
                                     (str(self.imlist[index])[:-4] + '.tif')), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        # m = cv2.imread(os.path.join(self.config.tmask_dir, self.config.mask_dir_name[0],
        #                             (str(self.imlist[index])[:-4] + '.bmp')), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        # m1 = cv2.imread(os.path.join(self.config.tmask_dir, self.config.mask_dir_name[1],
        #                              (str(self.imlist[index])[:-4] + '.bmp')), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        # m2 = cv2.imread(os.path.join(self.config.tmask_dir, self.config.mask_dir_name[2],
        #                              (str(self.imlist[index])[:-4] + '.bmp')), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        #将图片大小改为128*128
        # cloud = cv2.resize(cloud, (128, 128), interpolation=cv2.INTER_CUBIC)
        # x1 = cv2.resize(x1, (128, 128), interpolation=cv2.INTER_CUBIC)
        # x2 = cv2.resize(x2, (128, 128), interpolation=cv2.INTER_CUBIC)
        # #将图片大小改为128*128
        # m = cv2.resize(m, (128, 128), interpolation=cv2.INTER_CUBIC)
        # m1 = cv2.resize(m1, (128, 128), interpolation=cv2.INTER_CUBIC)
        # m2 = cv2.resize(m2, (128, 128), interpolation=cv2.INTER_CUBIC)
        m = 1 - (m / 255)
        m1 = 1 - (m1 / 255)
        m2 = 1 - (m2 / 255)

        # 归一化
        temp = np.dstack((cloud, x1, x2)) / 255
        m = np.dstack((m, m1, m2))
        # augments = self.transform(image=temp, mask=m)
        gt1 = torch.from_numpy(np.transpose(temp, (2, 0, 1))[0:1, :, :])
        gt2 = torch.from_numpy(np.transpose(temp, (2, 0, 1))[1:2, :, :])
        gt3 = torch.from_numpy(np.transpose(temp, (2, 0, 1))[2:3, :, :])
        gt = [gt1, gt2, gt3]
        mask1 = torch.from_numpy(np.transpose(m, (2, 0, 1))[0:1, :, :])
        mask2 = torch.from_numpy(np.transpose(m, (2, 0, 1))[1:2, :, :])
        mask3 = torch.from_numpy(np.transpose(m, (2, 0, 1))[2:3, :, :])
        m = [mask1, mask2, mask3]
        name_ = str(self.imlist[index])[:-4] + '.tif'
        if self.is_real:
            name_ = str(self.imlist[index])[:-4] + '.tif'
        return gt, m, name_, self.config.cloud[0], self.config.cloud[1], self.config.cloud[2]

    def __len__(self):
        return len(self.imlist)