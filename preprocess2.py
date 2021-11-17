import cv2
import os
import numpy as np
import nibabel as nib
import collections

DATA_ROOT = '/data/ffy/dataset/MICCAI_BraTS_2019_Data_Training'
OUTPUT_ROOT = './train'
HGG_ROOT = os.path.join(OUTPUT_ROOT, 'HGG')
LGG_ROOT = os.path.join(OUTPUT_ROOT, 'LGG')

# MRI Image channels Description
# ch0: FLAIR / ch1: T1 / ch2: T2/ ch3: T1CE
# Ground Truth channels Description
# ch0: WT / ch1: TC / ch2: ET

def normalize(slice, bottom=99, down=1):
    """
    normalize image with mean and std for regionnonzero,and clip the value into range
    :param slice:
    :param bottom:
    :param down:
    :return:
    """
    #有点像“去掉最低分去掉最高分”的意思,使得数据集更加“公平”
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)#限定范围numpy.clip(a, a_min, a_max, out=None)

    #除了黑色背景外的区域要进行标准化
    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        # since the range of intensities is between 0 and 5000 ,
        # the min in the normalized slice corresponds to 0 intensity in unnormalized slice
        # the min is replaced with -9 just to keep track of 0 intensities
        # so that we can discard those intensities afterwards when sampling random patches
        tmp[tmp == tmp.min()] = -9 #黑色背景区域
        return tmp

if __name__ == "__main__":
    try:
        os.mkdir(OUTPUT_ROOT)
    except:
        pass
    try:
        os.mkdir(HGG_ROOT)
    except:
        pass
    try:
        os.mkdir(LGG_ROOT)
    except:
        pass

    # LGG Images
    img_output_path = os.path.join(LGG_ROOT, 'image')
    try:
        os.mkdir(img_output_path)
    except:
        pass
    label_output_path = os.path.join(LGG_ROOT, 'label')
    try:
        os.mkdir(label_output_path)
    except:
        pass
    data_path = os.path.join(DATA_ROOT, 'LGG')
    for img_name in os.listdir(data_path):
        img_path = os.path.join(data_path, img_name)
        modals_file_list = os.listdir(img_path)
        modals_dict = {}
        for i,val in enumerate(modals_file_list):
            modals_dict[val.split('.')[0].split('_')[-1]] = modals_file_list[i]
        print(modals_dict)
        for i,key in enumerate(modals_dict):
            modal_path = os.path.join(img_path, modals_dict[key])
            modal = nib.load(modal_path)
            modal = (modal.get_fdata())[:, :, :]
            if key=='seg':
                seg = modal.astype(np.uint8)
            else:
                print(np.bincount(modal.astype(np.uint8).flatten()))
                modal = (modal / modal.max()) * 255
                if key=='flair':
                    flair = modal.astype(np.uint8)
                elif key=='t1':
                    t1 = modal.astype(np.uint8)
                elif key=='t2':
                    t2 = modal.astype(np.uint8)
                elif key=='t1ce':
                    t1ce = modal.astype(np.uint8)
        for i in range(modal.shape[2]):
            fn = os.path.join(label_output_path, img_name + '_' + str(i) + '.tiff')
            label = np.array(seg[:, :, i])
            label = np.tile(label, (3, 1, 1))
            for i in range(label.shape[0]):
                if i == 0:
                    for x in np.nditer(label[i], op_flags=['readwrite']):
                        if x == 4:
                            x[...] = 255
                        else:
                            x[...] = 0
                elif i == 1:
                    for x in np.nditer(label[i], op_flags=['readwrite']):
                        if x == 4 or x == 1:
                            x[...] = 255
                        else:
                            x[...] = 0
                elif i == 2:
                    for x in np.nditer(label[i], op_flags=['readwrite']):
                        if x == 0:
                            x[...] = 0
                        else:
                            x[...] = 255
            label = np.array(label).transpose((1,2,0))
            #cv2.imwrite(fn, label)
            fn = os.path.join(img_output_path, img_name + '_' + str(i) + '.tiff')
            image = [flair[:, :, i], t1[:, :, i], t1ce[:, :, i], t2[:, :, i]]
            image = np.array(image).transpose((1,2,0))
            #print(image.shape)
            #cv2.imwrite(fn, image)

    # HGG Images
    img_output_path = os.path.join(HGG_ROOT, 'image')
    try:
        os.mkdir(img_output_path)
    except:
        pass
    label_output_path = os.path.join(HGG_ROOT, 'label')
    try:
        os.mkdir(label_output_path)
    except:
        pass
    data_path = os.path.join(DATA_ROOT, 'HGG')
    for img_name in os.listdir(data_path):
        img_path = os.path.join(data_path, img_name)
        modals_file_list = os.listdir(img_path)
        modals_dict = {}
        for i,val in enumerate(modals_file_list):
            modals_dict[val.split('.')[0].split('_')[-1]] = modals_file_list[i]
        print(modals_dict)
        for i,key in enumerate(modals_dict):
            modal_path = os.path.join(img_path, modals_dict[key])
            modal = nib.load(modal_path)
            modal = (modal.get_fdata())[:, :, :]
            if key=='seg':
                seg = modal.astype(np.uint8)
            else:

                modal = (modal / modal.max()) * 255
                if key=='flair':
                    flair = modal.astype(np.uint8)
                elif key=='t1':
                    t1 = modal.astype(np.uint8)
                elif key=='t2':
                    t2 = modal.astype(np.uint8)
                elif key=='t1ce':
                    t1ce = modal.astype(np.uint8)
        for i in range(modal.shape[2]):
            fn = os.path.join(label_output_path, img_name + '_' + str(i) + '.tiff')
            label = np.array(seg[:, :, i])
            label = np.tile(label, (3, 1, 1))
            for i in range(label.shape[0]):
                if i == 0:
                    for x in np.nditer(label[i], op_flags=['readwrite']):
                        if x == 4:
                            x[...] = 255
                        else:
                            x[...] = 0
                elif i == 1:
                    for x in np.nditer(label[i], op_flags=['readwrite']):
                        if x == 4 or x == 1:
                            x[...] = 255
                        else:
                            x[...] = 0
                elif i == 2:
                    for x in np.nditer(label[i], op_flags=['readwrite']):
                        if x == 0:
                            x[...] = 0
                        else:
                            x[...] = 255
            label = np.array(label).transpose((1,2,0))
            #cv2.imwrite(fn, label)
            fn = os.path.join(img_output_path, img_name + '_' + str(i) + '.tiff')
            image = [flair[:, :, i], t1[:, :, i], t1ce[:, :, i], t2[:, :, i]]
            image = np.array(image).transpose((1,2,0))
            #print(image.shape)
            #cv2.imwrite(fn, image)