import glob
import cv2
import numpy as np


def calculate_mean_ccm():
    # mode = 'night'
    mode = 'default'
    semFolder = 'ccm_data/match_input_%s' % mode
    semList = glob.glob(semFolder + '/*.png')
    gtFolder = 'ccm_data/match_ref_%s' % mode
    gtList = glob.glob(gtFolder + '/*.jpg')
    num_img = len(semList)
    i = 0
    G = np.zeros((num_img, 3, 3))
    for semname, gtname in zip(semList, gtList):
        print(i)
        sem = cv2.imread(semname, -1)
        height, width, ch = sem.shape
        sem = (sem/255.0).reshape(-1, 3).T
        gt = cv2.imread(gtname, -1)
        gt = cv2.resize(gt, (width, height))
        gt = (gt/255.0).reshape(-1, 3).T
        G[i] = np.matmul(gt, np.linalg.pinv(sem))
        i += 1
    # import pdb; pdb.set_trace()

    G_mean = np.mean(G, axis=0)
    semFolder = '/data1/front_cam/test_ccm_cal/all_input'
    semList = glob.glob(semFolder + '/*.png')
    i = 0
    for semname in semList:
        print(i)
        sem = cv2.imread(semname, -1)
        height, width, ch = sem.shape
        sem = (sem/255.0).reshape(-1, 3).T
        out = G_mean @ sem
        out = np.clip(out * 255.0, 0, 255)
        out = out.T.reshape(height, width, ch).astype(np.uint8)
        cv2.imwrite(semname.replace(
            'all_input', 'matched_input_%s' % mode), out)
        # cv2.imwrite(semname.replace('input', 'input_corrected_default'), out)
        i += 1
        # import pdb; pdb.set_trace()


if __name__ == "__main__":
    calculate_mean_ccm()
