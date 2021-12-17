import torch
import torch.utils.data as data

import numpy as np
import glob
import cv2


def get_image_index(path):
    return int(path.split('.')[0].split('/')[-1])


def populate_train_list(images_path):

    # gt_list = [images_path + '/6G7M_20150325_113420_008/resized_final.png'] *5000
    # input_list = [images_path + '/6G7M_20150325_113420_008/resized_input_merged.png'] *5000

    gtfolders = glob.glob(images_path + '/*')
    gt_list = []
    sizemismatch_list = ['data/hdrplus/20171106_subset/results_20171023/0127_20161107_171749_524',
                         'data/hdrplus/20171106_subset/results_20171023/9bf4_20150824_210544_967',
                         'data/hdrplus/20171106_subset/results_20171023/c483_20150901_105412_265']
    for folder in gtfolders:
        if folder in sizemismatch_list:
            print('input and gt size mismatch: %s' % folder)
            continue
        gt_list.append(folder + '/input_merged.png')
    # gt_list.sort()

    input_list = map(
        lambda x: x.replace('input_merged', 'input'),
        gt_list)

    return list(input_list), list(gt_list)


def augment_patch(patch, mode):
    # rotate counterclockwise 0,1,2,3 times, flip updown 0,1 times
    patch = np.rot90(patch, k=mode % 4)
    if mode > 3:
        patch = np.flipud(patch)
    return patch


class DenoiseDataset(data.Dataset):

    def __init__(self, images_path, size, phase, num_patch=64):

        self.input_list, self.gt_list = populate_train_list(images_path)
        self.size = size  # patch/image size
        self.phase = phase
        self.num_patch = num_patch
        # np.random.seed(0)
        # import pdb; pdb.set_trace()
        print("Total training images:", len(self.input_list))

    def __getitem__(self, index):
        # print(index)
        image_input = cv2.imread(self.input_list[index], cv2.IMREAD_UNCHANGED)
        image_gt = cv2.imread(self.gt_list[index], cv2.IMREAD_UNCHANGED)

        h, w, ch = image_input.shape
        h2, w2, ch2 = image_gt.shape
        # if h!=h2 or w!=w2:
        # 	print('input and gt size mismatch: %s'%self.input_list[index])
        # 	h = min(h, h2)
        # 	w = min(w, w2)
        if self.phase == 'train':
            data_input = np.zeros(
                (self.num_patch, self.size, self.size, ch), dtype=image_input.dtype)
            data_gt = np.zeros(
                (self.num_patch, self.size, self.size, ch), dtype=image_input.dtype)
            for i in range(self.num_patch):
                # get a random patch of the image
                randh0 = np.random.randint(0, max(0, h - self.size))
                randw0 = np.random.randint(0, max(0, w - self.size))
                patch_input = image_input[randh0:randh0 +
                                          self.size, randw0:randw0+self.size, :]
                patch_gt = image_gt[randh0:randh0 +
                                    self.size, randw0:randw0+self.size, :]
                # augment - flip, rotate
                mode = np.random.randint(0, 8)  # [0,8) modes, rot4*flip2
                patch_input = augment_patch(patch_input, mode)
                patch_gt = augment_patch(patch_gt, mode)
                data_input[i] = patch_input
                data_gt[i] = patch_gt

            data_input = (np.asarray(data_input[..., ::-1])/65535.0)
            data_input = torch.from_numpy(data_input).float()
            data_input = data_input.permute(0, 3, 1, 2)
            data_gt = (np.asarray(data_gt[..., ::-1])/65535.0)
            data_gt = torch.from_numpy(data_gt).float()
            data_gt = data_gt.permute(0, 3, 1, 2)
        else:
            # get the entire input image (cropped to square)
            randh0 = 0
            randw0 = 0
            data_input = image_input[randh0:randh0 +
                                     self.size, randw0:randw0+self.size, :]
            data_gt = image_gt[randh0:randh0 +
                               self.size, randw0:randw0+self.size, :]

            data_input = (np.asarray(data_input[..., ::-1])/65535.0)
            data_input = torch.from_numpy(data_input).float()
            data_input = data_input.permute(2, 0, 1)
            data_gt = (np.asarray(data_gt[..., ::-1])/65535.0)
            data_gt = torch.from_numpy(data_gt).float()
            data_gt = data_gt.permute(2, 0, 1)

        return data_input, data_gt

    def __len__(self):
        return len(self.input_list)
