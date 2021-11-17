import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.utils as vutils
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
import collections


class BraTS(Dataset):
    def __init__(self, path, mode='train'):
        self.mode = mode
        self.path = path

        '''load training data'''
        self.dir_img = os.path.join(path, 'image')
        self.dir_gt = os.path.join(path, 'label')
        self.img = []
        self.gt = []
        for _, name in enumerate(os.listdir(self.dir_img)):
            nb = int(name.split('/')[-1].split('.')[0].split('_')[-1])
            if 10 <= nb < 140:
                self.img.append(os.path.join(self.dir_img, name))
                self.gt.append(os.path.join(self.dir_gt, name))

    def __getitem__(self, item):
        fn = self.img[item].split('/')[-1].split('.')[0]
        img = Image.open(self.img[item])
        width = img.size[0]
        delta_w = width - 200
        left_d = int(delta_w / 2)
        right_d = width - (delta_w - left_d)
        img = img.crop([left_d, left_d, right_d, right_d])
        img = transforms.ToTensor()(img)
        img = transforms.Normalize((0.08412773392539548, 0.11307919895094626, 0.08010867650885856, 0.06903648909524748),
                                   (0.15437624225813634, 0.19926509643706336, 0.14870959466563555, 0.1219899915268555))(img)
        gt = Image.open(self.gt[item])
        gt = gt.crop([left_d, left_d, right_d, right_d])
        #d = collections.Counter(np.array(gt).flatten())
        #for k in d:
        #    print(k)
        gt = transforms.ToTensor()(gt)
        return img, gt, fn

    def __len__(self):
        return len(self.img)


if __name__ == '__main__':
    hgg = '../train/HGG'
    lgg = '../train/LGG'
    train_data = BraTS(path=hgg, mode='train') + BraTS(path=lgg, mode='train')
    train_loader = DataLoader(dataset=train_data, batch_size=1)
    device = torch.device("cuda")

    print(len(train_data))

    # Plot some training images
    it = iter(train_loader)
    real_batch = next(it)
    real_batch = next(it)
    real_batch = next(it)
    real_batch = next(it)
    real_batch = next(it)
    real_batch = next(it)
    real_batch = next(it)
    real_batch = next(it)
    real_batch = next(it)
    print(real_batch[2])
    x = real_batch[0].to(device)[:64]
    x0, x1, x2, x3 = x.split(1, 1)
    y = real_batch[1].to(device)[:64]
    y0, y1, y2 = y.split(1, 1)
    """
    y0 = np.array(y0.cpu()) * 255
    y0 = y0.astype('uint8')[0,:].reshape(200,200,1)
    print(collections.Counter(y0.flatten()))
    _, y0 = cv2.threshold(y0, 127, 255, cv2.THRESH_BINARY)
    _, y0, _ = cv2.findContours(y0, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    y0 = np.array(y0)
    print(y0)
    print(collections.Counter(y0))
    y0 = transforms.ToTensor()(y0).unsqueeze(0)
    print(y0.shape)
    """
    # subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(2, 4)
    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    axarr[0,0].imshow(np.transpose(vutils.make_grid(x0, padding=2, normalize=True).cpu(), (1, 2, 0)))
    axarr[0,1].imshow(np.transpose(vutils.make_grid(x1, padding=2, normalize=True).cpu(), (1, 2, 0)))
    axarr[0,2].imshow(np.transpose(vutils.make_grid(x2, padding=2, normalize=True).cpu(), (1, 2, 0)))
    axarr[0,3].imshow(np.transpose(vutils.make_grid(x3, padding=2, normalize=True).cpu(), (1, 2, 0)))
    axarr[1,0].imshow(np.transpose(vutils.make_grid(y0, padding=2, normalize=True).cpu(), (1, 2, 0)))
    axarr[1,1].imshow(np.transpose(vutils.make_grid(y1, padding=2, normalize=True).cpu(), (1, 2, 0)))
    axarr[1,2].imshow(np.transpose(vutils.make_grid(y2, padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.suptitle("Loaded Images")
    plt.show()


