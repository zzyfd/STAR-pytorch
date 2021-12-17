import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2
import os, shutil


# random.seed(1143)

def div_data(img_path):
    input_list = glob.glob(os.path.abspath(img_path + '/Inputs_jpg/*'))
    input_val = random.sample(input_list, len(input_list) // 10)
    input_train = list(set(input_list) - set(input_val))
    try:
        os.makedirs(img_path + '/Inputs_jpg_val/')
        os.makedirs(img_path + '/Experts_C_val/')
        os.makedirs(img_path + '/Inputs_jpg_train/')
        os.makedirs(img_path + '/Experts_C_train/')
    except:
        pass
    for data in input_val:
        shutil.copy(data, img_path + '/Inputs_jpg_val/')
        shutil.copy(data.replace('Inputs_jpg', 'Experts_C'), img_path + '/Experts_C_val/')
    for data in input_train:
        shutil.copy(data, img_path + '/Inputs_jpg_train/')
        shutil.copy(data.replace('Inputs_jpg', 'Experts_C'), img_path + '/Experts_C_train/')

    return


def get_image_index(path):
    return int(path.split('.')[0].split('/')[-1])


def populate_train_list(images_path):
    input_list = glob.glob(os.path.abspath(images_path + '/Inputs_jpg/*'))

    gt_list = map(
        lambda x: x.replace('Inputs_jpg', 'Experts_C'),
        input_list)

    return list(input_list), list(gt_list)


def RGBtoYUV444(rgb):
    # code from Jun
    # yuv range: y[0,1], uv[-0.5, 0.5]
    height, width, ch = rgb.shape
    assert ch == 3, 'rgb should have 3 channels'

    rgb2yuv_mat = np.array([[0.299, 0.587, 0.114], [-0.16874, -
    0.33126, 0.5], [0.5, -0.41869, -0.08131]], dtype=np.float32)

    rgb_t = rgb.transpose(2, 0, 1).reshape(3, -1)
    yuv = rgb2yuv_mat @ rgb_t
    yuv = yuv.transpose().reshape((height, width, 3))

    # return yuv.astype(np.float32)
    # rescale uv to [0,1]
    yuv[:, :, 1] += 0.5
    yuv[:, :, 2] += 0.5
    return yuv


class EnhanceDataset_FiveK(data.Dataset):

    def __init__(self, images_path, image_size, image_size_w = None,is_yuv=False, resize=True):

        self.input_list, self.gt_list = populate_train_list(
            images_path)
        self.image_size = image_size
        image_size_w = image_size if image_size_w is None else image_size_w
        self.image_size_w = image_size_w
        self.resize = resize
        self.is_yuv = is_yuv
        # import pdb; pdb.set_trace()
        print("Total training examples:", len(self.input_list))

    def __getitem__(self, index):
        # print(index)
        data_input = cv2.imread(self.input_list[index], cv2.IMREAD_UNCHANGED)
        data_gt = cv2.imread(self.gt_list[index], cv2.IMREAD_UNCHANGED)

        if data_input.shape[0] >= data_input.shape[1]:
            data_input = cv2.transpose(data_input)
            data_gt = cv2.transpose(data_gt)

        if self.resize:
            data_gt = cv2.resize(data_gt, (self.image_size, self.image_size_w))
            data_input = cv2.resize(data_input, (self.image_size, self.image_size_w))




        data_input = (np.asarray(data_input[..., ::-1]) / 255.0)
        if self.is_yuv:
            data_input = RGBtoYUV444(data_input)
        data_input = torch.from_numpy(data_input).float()  # float32
        data_gt = (np.asarray(data_gt[..., ::-1]) / 255.0)
        if self.is_yuv:
            data_gt = RGBtoYUV444(data_gt)
        data_gt = torch.from_numpy(data_gt).float()
        return data_input.permute(2, 0, 1), data_gt.permute(2, 0, 1)

    def __len__(self):
        return len(self.input_list)


if __name__ == '__main__':
    img_path = 'G:\\fivek_dataset\\fivek_dataset\\raw_photos\\FiveK_Lightroom_Export_InputDayLight'
    li = div_data(img_path)
    print(li)
