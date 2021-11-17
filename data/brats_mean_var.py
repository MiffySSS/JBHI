import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.utils as vutils
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import collections
from imageio import imread


if __name__ == '__main__':
    path = '../train'
    '''load training data'''
    dir_hgg = os.path.join(path, 'HGG')
    dir_lgg = os.path.join(path, 'LGG')
    img = []
    gt = []
    # HGG data
    dir_img = os.path.join(dir_hgg, 'image')
    dir_gt = os.path.join(dir_hgg, 'label')
    for _, name in enumerate(os.listdir(dir_img)):
        nb = int(name.split('/')[-1].split('.')[0].split('_')[-1])
        if 10 <= nb < 140:
            img.append(os.path.join(dir_img, name))
            gt.append(os.path.join(dir_gt, name))
    # LGG data
    dir_img = os.path.join(dir_lgg, 'image')
    dir_gt = os.path.join(dir_lgg, 'label')
    for _, name in enumerate(os.listdir(dir_img)):
        nb = int(name.split('/')[-1].split('.')[0].split('_')[-1])
        if 10 <= nb < 140:
            img.append(os.path.join(dir_img, name))
            gt.append(os.path.join(dir_gt, name))

    print(len(img))

    flair = 0
    t1 = 0
    t1ce = 0
    t2 = 0
    for idx in range(len(img)):
        fn = img[idx]
        data = np.array(Image.open(fn).crop([20, 20, 220, 220])) / 255.0
        flair += np.sum(data[:,:,0])
        t1    += np.sum(data[:,:,1])
        t1ce  += np.sum(data[:,:,2])
        t2    += np.sum(data[:,:,3])

    num = len(img) * 200 * 200
    flair_mean = flair / num
    t1_mean    = t1    / num
    t1ce_mean  = t1ce  / num
    t2_mean    = t2    / num
    print('mean: ', flair_mean, t1_mean, t1ce_mean, t2_mean)

    flair = 0
    t1 = 0
    t1ce = 0
    t2 = 0
    for idx in range(len(img)):
        fn = img[idx]
        data = np.array(Image.open(fn).crop([20, 20, 220, 220])) / 255.0
        flair += np.sum((data[:,:,0] - flair_mean) ** 2)
        t1    += np.sum((data[:,:,1] - t1_mean)    ** 2)
        t1ce  += np.sum((data[:,:,2] - t1ce_mean)  ** 2)
        t2    += np.sum((data[:,:,3] - t2_mean)    ** 2)

    flair_var = np.sqrt(flair / num)
    t1_var    = np.sqrt(t1 / num)
    t1ce_var  = np.sqrt(t1ce / num)
    t2_var    = np.sqrt(t2 / num)
    print('var: ', flair_var, t1_var, t1ce_var, t2_var)