import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2

# random.seed(1143)


def get_image_index(path):
    return int(path.split('.')[0].split('/')[-1])


def populate_train_list(images_path):

    gt_list = glob.glob(images_path + '/*_resized_final.png')

    input_list = map(
        lambda x: x.replace('resized_final', 'resized_input_merged_corrected'),
        gt_list)

    matte_list = map(
        lambda x: x.replace('resized_final', 'resized_mask'),
        gt_list)

    return list(input_list), list(gt_list), list(matte_list)


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


class EnhanceDataset(data.Dataset):

    def __init__(self, images_path, image_size, is_yuv=False, resize=False):

        self.input_list, self.gt_list, self.matte_list = populate_train_list(
            images_path)
        self.image_size = image_size
        self.is_yuv = is_yuv
        # import pdb; pdb.set_trace()
        print("Total training examples:", len(self.input_list))

    def __getitem__(self, index):
        # print(index)
        data_input = cv2.imread(self.input_list[index], cv2.IMREAD_UNCHANGED)
        # data_input = cv2.resize(data_input, (self.image_size,self.image_size))
        data_input = (np.asarray(data_input[..., ::-1])/65535.0)
        if self.is_yuv:
            data_input = RGBtoYUV444(data_input)
        data_input = torch.from_numpy(data_input).float()  # float32

        data_gt = cv2.imread(self.gt_list[index], cv2.IMREAD_UNCHANGED)
        # data_gt = cv2.resize(data_gt, (self.image_size,self.image_size))
        data_gt = (np.asarray(data_gt[..., ::-1])/255.0)
        if self.is_yuv:
            data_gt = RGBtoYUV444(data_gt)
        data_gt = torch.from_numpy(data_gt).float()

        data_matte = cv2.imread(self.matte_list[index], cv2.IMREAD_UNCHANGED)
        # data_gt = cv2.resize(data_gt, (self.image_size,self.image_size))
        data_matte = (np.asarray(data_matte)/255.0)
        data_matte = torch.from_numpy(data_matte).float()
        data_matte = torch.unsqueeze(data_matte, -1)

        return data_input.permute(2, 0, 1), data_gt.permute(2, 0, 1), data_matte.permute(2, 0, 1)

    def __len__(self):
        return len(self.input_list)
