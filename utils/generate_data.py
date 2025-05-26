import os
import sys
from osgeo import gdal
import cv2
from PIL import Image
from attrdict import AttrMap
import yaml
from utils import read_simple_tif, mkdir
import numpy as np
Image.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
item_width = 256
item_height = 256
from scipy import misc

def generateMask(inpath, width, height):
    ds = gdal.Open(inpath)
    # 判断是否读取到数据
    if ds is None:
        print('Unable to open *.tif')
        sys.exit(1)  # 退出
    dt = ds.GetRasterBand(1)
    data = dt.ReadAsArray()

    del ds
    mask1 = data[0:height, 0:width]
    mask2 = data[0:height, width:width * 2]
    mask3 = data[height:height * 2, 0:width]
    mask4 = data[height:height * 2, width:width * 2]

    return [mask1, mask1, mask1, mask1]


if __name__ == '__main__':
    with open('../config.yml', 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = AttrMap(config)

    # path = r'H:\landsat5'
    pathlist = [r'H:\去云\武汉地区\cut\test']
    years = ['20130512', '20130613', '20130731', '20130917', '20131120']
    bands = ['B2', 'B3', 'B4', 'B5']
    for year in years:
        result = r'E:\cloudremove\dataset\\test\\' + year
        mkdir(result)
        index = 0
        for band in bands:
            for i, path in enumerate(pathlist):
                n = 0
                if i == 0:
                    name_ = path + '\\' + year + '_T_' + band + '.TIF'
                else:
                    name_ = path + '\\' + year + '_S_' + band + '.TIF'
                tdata = read_simple_tif(name_)
                height, width = tdata.shape
                for i in range(0, int(height / item_height)):
                    for j in range(0, int(width / item_width)):
                        cropped = tdata[i * item_height:(i + 1) * item_height, j * item_width:(j + 1) * item_width]
                        cv2.imwrite(result + '/' + str(index) + '.tif', cropped)
                        index = index + 1
        print('数据集大小为{}'.format(index))

    # index = 0
    # result_ = r'E:\PSTCR\data\\' + 'mask'
    # mkdir(result_)
    # height, width = masks[0].shape
    # for mask in masks:
    #     for i in range(0, int(height / item_height)):
    #         for j in range(0, int(width / item_width)):
    #             cropped = mask[i * item_height:(i + 1) * item_height, j * item_width:(j + 1) * item_width]
    #             cv2.imwrite(result_ + '\\' + str(index) + '.tif', cropped)
    #             index = index + 1
    #
    # height, width = masks_[0].shape
    # for mask_ in masks_:
    #     for i in range(0, int(height / item_height)):
    #         for j in range(0, int(width / item_width)):
    #             cropped = mask_[i * item_height:(i + 1) * item_height, j * item_width:(j + 1) * item_width]
    #             cv2.imwrite(result_ + '\\' + str(index) + '.tif', cropped)
    #             index = index + 1
    # print('掩膜数据集大小为{}'.format(index))


    # k = 0
    # figures = os.listdir(r"H:\去云\landsat-5-b432-全幅")
    # for figure in figures:
    #     result = r'H:\去云\\' + figure[0:8]
    #     mkdir(result)
    #     index = 0
    #     name_ = r"H:\去云\landsat-5-b432-全幅\\" + figure
    #     dataset = gdal.Open(name_)
    #     tdata = dataset.ReadAsArray().transpose(1, 2, 0)
    #     # tdata = read_simple_tif(name_)
    #     height, width, _ = tdata.shape
    #     for i in range(0, int(height / item_height)):
    #         for j in range(0, int(width / item_width)):
    #             cropped = tdata[i * item_height:(i + 1) * item_height, j * item_width:(j + 1) * item_width]
    #             # driver = gdal.GetDriverByName("GTiff")
    #             # New_YG_dataset = driver.Create(result + '\\' + str(index) + '.tif', item_width, item_height, 3, gdal.GDT_Float32)
    #             # New_YG_dataset.GetRasterBand(1).WriteArray(cropped[0])
    #             # New_YG_dataset.GetRasterBand(2).WriteArray(cropped[1])
    #             # New_YG_dataset.GetRasterBand(3).WriteArray(cropped[2])
    #             # New_YG_dataset.FlushCache()
    #             cropped = np.array(cropped)
    #             img = Image.fromarray(cropped).convert('RGB')
    #             img.save(result + '\\' + str(index) + '.bmp')
    #             # misc.imsave(result + '\\' + str(index) + '.png', cropped)
    #             # cv2.imwrite(result + '\\' + str(index) + '.png', cropped)
    #             index = index + 1




    # for year in years:
    #     result = r'H:\去云\\' + year
    #     mkdir(result)
    #     index = 0
    #     for path in pathlist:
    #         n = 0
    #         name_ = path + '\\' + year + '_01_T1_' + band + '.TIF'
    #         tdata = read_simple_tif(name_)
    #         height, width = tdata.shape
    #         for i in range(0, int(height / item_height)):
    #             for j in range(0, int(width / item_width)):
    #                 cropped = tdata[i * item_height:(i + 1) * item_height, j * item_width:(j + 1) * item_width]
    #                 cv2.imwrite(result + '/' + str(index) + '.tif', cropped)
    #                 index = index + 1
    #     print('数据集大小为{}'.format(index))
