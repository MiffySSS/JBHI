# JBHI-Pytorch

This repository contains a reference implementation of the algorithms described in our paper ["Self-supervised Multi-modal Hybrid Fusion Network for Brain Tumor Segmentation"](https://ieeexplore.ieee.org/abstract/document/9529036/).

## Introduction

In this paper, we propose a multi-modal brain tumor segmentation framework that adopts the hybrid fusion of modality-specific features using a self-supervised learning strategy. The algorithm is based on a fully convolutional neural network. Firstly, we propose a multi-input architecture that learns independent features from multi-modal data, and can be adapted to different numbers of multi-modal inputs. Compared with single-modal multi-channel networks, our model provides a better feature extractor for segmentation tasks, which learns cross-modal information from multi-modal data. Secondly, we propose a new feature fusion scheme, named hybrid attentional fusion. This scheme enables the network to learn the hybrid representation of multiple features and capture the correlation information between them through an attention mechanism. Unlike popular methods, such as feature map concatenation, this scheme focuses on the complementarity between multi-modal data, which can significantly improve the segmentation results of specific regions. Thirdly, we propose a self-supervised learning strategy for brain tumor segmentation tasks. Our experimental results demonstrate the effectiveness of the proposed model against other state-of-the-art multi-modal medical segmentation methods.


## Dataset

-BraTS19. Download [here](https://pan.baidu.com/s/1S5XGTdHkwFnagKS-5vWYBg#list/path=%2F) code:2333.

## Quick Start.

### Install Dependencies

- python=3.6
- PyTorch=1.3.1
- numpy
- scipy
- SimpleITK

### Database Preprocessing

```
cd data && python preprocess_example.py
```

### Training Demo

```
python train.py --model multi_modal_seg --dataset brats --backbone resnet50 --lr 1e-2 --epoch 50 --batch-size 16
```

### Pretrained Models

```
coming soon...
```
