import numpy as np
import cv2
import glob
import random
from skimage.io import imread
from skimage import color
import torch
import torch.utils.data
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt


class BraTS_new(Dataset):

    def __init__(self, img_paths, mask_paths):
        self.img_paths = img_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        #读numpy数据(npy)的代码
        npimage = np.load(img_path)
        npmask = np.load(mask_path)
        npimage = npimage.transpose((2, 0, 1))

        WT_Label = npmask.copy()
        WT_Label[npmask == 1] = 1.
        WT_Label[npmask == 2] = 1.
        WT_Label[npmask == 4] = 1.
        TC_Label = npmask.copy()
        TC_Label[npmask == 1] = 1.
        TC_Label[npmask == 2] = 0.
        TC_Label[npmask == 4] = 1.
        ET_Label = npmask.copy()
        ET_Label[npmask == 1] = 0.
        ET_Label[npmask == 2] = 0.
        ET_Label[npmask == 4] = 1.
        nplabel = np.empty((160, 160, 3))
        nplabel[:, :, 0] = WT_Label
        nplabel[:, :, 1] = TC_Label
        nplabel[:, :, 2] = ET_Label
        nplabel = nplabel.transpose((2, 0, 1))

        nplabel = nplabel.astype("float32")
        npimage = npimage.astype("float32")

        return npimage,nplabel


if __name__ == '__main__':
    train_img_paths = glob.glob("./trainImage/*")
    val_img_paths = glob.glob("./trainGt/*")
    print("train_num:%s" % str(len(train_img_paths)))
    train_data = BraTS_new(img_paths=train_img_paths, mask_paths=val_img_paths)
    train_loader = DataLoader(dataset=train_data, batch_size=1)
    device = torch.device("cuda")

    print(len(train_data))

    # Plot some training images
    it = iter(train_loader)
    real_batch = next(it)
    real_batch = next(it)
    real_batch = next(it)
    #real_batch = next(it)
    real_batch = next(it)
    real_batch = next(it)
    #real_batch = next(it)
    #real_batch = next(it)
    real_batch = next(it)
    x = real_batch[0].to(device)[:64]
    x0, x1, x2, x3 = x.split(1, 1)
    y = real_batch[1].to(device)[:64]
    y0, y1, y2 = y.split(1, 1)
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
