import time
import datetime
import os
import shutil
import sys
from tqdm import tqdm
import glob

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision import utils
from tensorboardX import SummaryWriter

from args_train import parse_args
from data import get_segmentation_dataset
import segmentation
import self_sup
import transformer
from kits import metrics
from kits import SegMetrics
from kits import configure_loss
from kits import generate_params
from kits import setup_logger
from kits import LR_Scheduler
from kits import Saver
from kits import TensorboardSummary
from kits.distributed import *


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda')

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        hgg_path = "./train/HGG"
        lgg_path = "./train/LGG"
        print('data in path:', hgg_path, lgg_path)
        train_img_paths = glob.glob("./data/trainImage/*")
        val_img_paths = glob.glob("./data/trainGt/*")
        whole_dataset = get_segmentation_dataset('new', img_paths=train_img_paths, mask_paths=val_img_paths)
        '''
        whole_dataset = get_segmentation_dataset(self.args.dataset, path=hgg_path) \
                        + get_segmentation_dataset(self.args.dataset, path=lgg_path)
        '''
        whole_size = len(whole_dataset)
        train_size = int(0.8 * whole_size)
        val_size = whole_size - train_size
        self.train_size = train_size
        self.val_size = val_size
        train_dataset, val_dataset=data.random_split(whole_dataset,[train_size,val_size])
        print('train size: ', train_size)
        print('val size: ', val_size)
        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=args.workers,
                                            shuffle=True,
                                            pin_memory=True,
                                            drop_last=True)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_size=args.batch_size,
                                          num_workers=args.workers,
                                          pin_memory=True)

        # Define network
        if self.args.modalities == 'all':
            if self.args.model == 'unet':
                self.model = segmentation.UNet(4, 3, backbone=self.args.backbone).to(self.device)
            elif self.args.model == 'munet':
                self.model = segmentation.MUNet(4, 3, backbone=self.args.backbone).to(self.device)
            elif self.args.model == 'deeplab':
                self.model = segmentation.DeepLab(4, 3, backbone=self.args.backbone).to(self.device)
            elif self.args.model == 'multi_modal_seg':
                self.model = self_sup.multi_modality_seg(4, 3, backbone=self.args.backbone).to(self.device)
            elif self.args.model == 'transformer':
                self.model = transformer.MedT(img_size=160, imgchan=4, num_classes=3).to(self.device)
            else:
                raise Exception('model non supportable')
        elif self.args.modalities == 'flair' or self.args.modalities == 't1' \
                or self.args.modalities == 't1ce' or self.args.modalities == 't2':
            print('modal:', self.args.model)
            if self.args.model == 'unet':
                self.model = segmentation.UNet(1, 3, backbone=self.args.backbone).to(self.device)
            elif self.args.model == 'munet':
                self.model = segmentation.MUNet(1, 3, backbone=self.args.backbone).to(self.device)
            elif self.args.model == 'deeplab':
                self.model = segmentation.DeepLab(1, 3, backbone=self.args.backbone).to(self.device)
            elif self.args.model == 'multi_modal_seg':
                self.model = self_sup.multi_modality_seg(1, 3, backbone=self.args.backbone).to(self.device)
            else:
                raise Exception('model non supportable')
        else:
            if self.args.model == 'unet':
                self.model = segmentation.UNet(2, 3, backbone=self.args.backbone).to(self.device)
            elif self.args.model == 'munet':
                self.model = segmentation.MUNet(2, 3, backbone=self.args.backbone).to(self.device)
            elif self.args.model == 'deeplab':
                self.model = segmentation.DeepLab(2, 3, backbone=self.args.backbone).to(self.device)
            elif self.args.model == 'multi_modal_seg':
                self.model = self_sup.multi_modality_seg(2, 3, backbone=self.args.backbone).to(self.device)
            else:
                raise Exception('model non supportable')
        # Define optimizer
        #train_params = [{'params': self.model.get_1x_lr_params(), 'lr': args.lr},
        #                {'params': self.model.get_10x_lr_params(), 'lr': args.lr * 10}]
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum,
        #                                 weight_decay=args.weight_decay, nesterov=args.nesterov)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Define criterion
        self.criterion1 = configure_loss('dice')
        self.criterion2 = configure_loss('bce')

        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader))

        # Define Evaluator
        self.evaluator1 = SegMetrics(2)
        self.evaluator2 = SegMetrics(2)
        self.evaluator3 = SegMetrics(2)

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            #if args.cuda:
            #   self.model.module.load_state_dict(checkpoint['state_dict'])
            #else:
            self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))



    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        for iteration, sample in enumerate(tbar):
            image = sample[0].to(self.device)
            target = sample[1].to(self.device)
            self.scheduler(self.optimizer, iteration, epoch, self.best_pred)
            self.optimizer.zero_grad()
            if self.args.modalities == 'all':
                output = self.model(image)
            elif self.args.modalities == 'flair':
                output = self.model(image[:, 0, None, :, :])
            elif self.args.modalities == 't1':
                output = self.model(image[:, 1, None, :, :])
            elif self.args.modalities == 't1ce':
                output = self.model(image[:, 2, None, :, :])
            elif self.args.modalities == 't2':
                output = self.model(image[:, 3, None, :, :])
            elif self.args.modalities == 't1ce+flair':
                output = self.model(torch.cat((image[:, 2, None, :, :], image[:, 0, None, :, :]), dim=1))
            elif self.args.modalities == 't1ce+t2':
                output = self.model(torch.cat((image[:, 2, None, :, :], image[:, 3, None, :, :]), dim=1))
            elif self.args.modalities == 't1ce+t1':
                output = self.model(torch.cat((image[:, 2, None, :, :], image[:, 1, None, :, :]), dim=1))
            else:
                raise RuntimeError("modalities error, chossen{}".format(self.args.modalities))
            output = torch.sigmoid(output)
            dice_loss = self.criterion1(output, target)
            bce_loss = self.criterion2(output, target)
            loss = dice_loss + 0.5 * bce_loss
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (iteration + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), iteration + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if iteration % (num_img_tr // 10) == 0:
                global_step = iteration + num_img_tr * epoch
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, iteration * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % (train_loss / (iteration+1)))

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def validation(self, epoch):
        self.model.eval()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        hausdorff_WT_sum = 0.0
        hausdorff_TC_sum = 0.0
        hausdorff_ET_sum = 0.0
        for iteration, sample in enumerate(tbar):
            image = sample[0].to(self.device)
            target = sample[1].to(self.device)
            fn = sample[2]
            with torch.no_grad():
                if self.args.modalities == 'all':
                    output = self.model(image)
                elif self.args.modalities == 'flair':
                    output = self.model(image[:, 0, None, :, :])
                elif self.args.modalities == 't1':
                    output = self.model(image[:, 1, None, :, :])
                elif self.args.modalities == 't1ce':
                    output = self.model(image[:, 2, None, :, :])
                elif self.args.modalities == 't2':
                    output = self.model(image[:, 3, None, :, :])
                elif self.args.modalities == 't1ce+flair':
                    output = self.model(torch.cat((image[:, 2, None, :, :], image[:, 0, None, :, :]), dim=1))
                elif self.args.modalities == 't1ce+t2':
                    output = self.model(torch.cat((image[:, 2, None, :, :], image[:, 3, None, :, :]), dim=1))
                elif self.args.modalities == 't1ce+t1':
                    output = self.model(torch.cat((image[:, 2, None, :, :], image[:, 1, None, :, :]), dim=1))
                else:
                    raise RuntimeError("modalities error, chossen{}".format(self.args.modalities))
            output = torch.sigmoid(output)
            dice_loss = self.criterion1(output, target)
            bce_loss = self.criterion2(output, target)
            loss = dice_loss + 0.5 * bce_loss
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (iteration + 1)))

            pred = (output > 0.5).float()
            for i in range(image.size(0)):
                show = utils.make_grid([image[i, 0, None, :, :], image[i, 1, None, :, :],
                                        image[i, 2, None, :, :], image[i, 3, None, :, :],
                                        target[i, 0, None, :, :], target[i, 1, None, :, :],
                                        target[i, 2, None, :, :], torch.zeros_like(target[i, 2, None, :, :]),
                                        pred[i, 0, None, :, :], pred[i, 1, None, :, :],
                                        pred[i, 2, None, :, :], torch.zeros_like(pred[i, 2, None, :, :])],
                                        nrow=4, padding=2, pad_value=1.)
                self.writer.add_image(fn[i], show, global_step=epoch)

            pred = pred.long().cpu()
            target = target.cpu()
            pred = pred.numpy()
            target = target.numpy()
            hausdorff_WT = metrics.hausdorff_95(pred[:,0,:,:], target[:,0,:,:])
            hausdorff_TC = metrics.hausdorff_95(pred[:,1,:,:], target[:,1,:,:])
            hausdorff_ET = metrics.hausdorff_95(pred[:,2,:,:], target[:,2,:,:])
            hausdorff_WT_sum += hausdorff_WT
            hausdorff_TC_sum += hausdorff_TC
            hausdorff_ET_sum += hausdorff_ET
            self.evaluator1.update(pred[:,0,None,:,:], target[:,0,None,:,:])
            self.evaluator2.update(pred[:,1,None,:,:], target[:,1,None,:,:])
            self.evaluator3.update(pred[:,2,None,:,:], target[:,2,None,:,:])

        # Fast test during the training
        dice_WT = self.evaluator1.dice()
        dice_TC = self.evaluator2.dice()
        dice_ET = self.evaluator3.dice()
        dice_avg = (dice_WT + dice_TC + dice_ET) / 3
        hausdorff_WT = hausdorff_WT_sum / (iteration+1)
        hausdorff_TC = hausdorff_TC_sum / (iteration+1)
        hausdorff_ET = hausdorff_ET_sum / (iteration+1)
        hausdorff_avg = (hausdorff_WT + hausdorff_TC + hausdorff_ET) / 3
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/dice_WT', dice_WT, epoch)
        self.writer.add_scalar('val/dice_TC', dice_TC, epoch)
        self.writer.add_scalar('val/dice_ET', dice_ET, epoch)
        self.writer.add_scalar('val/dice_avg', dice_avg, epoch)
        self.writer.add_scalar('val/hausdorff_WT', hausdorff_WT, epoch)
        self.writer.add_scalar('val/hausdorff_TC', hausdorff_TC, epoch)
        self.writer.add_scalar('val/hausdorff_ET', hausdorff_ET, epoch)
        self.writer.add_scalar('val/hausdorff_avg', hausdorff_avg, epoch)

        sensitivity_WT = self.evaluator1.sensitivity()
        sensitivity_TC = self.evaluator2.sensitivity()
        sensitivity_ET = self.evaluator3.sensitivity()
        sensitivity_avg = (sensitivity_ET + sensitivity_TC + sensitivity_ET) / 3
        specificity_WT = self.evaluator1.specificity()
        specificity_TC = self.evaluator2.specificity()
        specificity_ET = self.evaluator3.specificity()
        specificity_avg = (specificity_WT + specificity_TC + specificity_ET) / 3
        self.writer.add_scalar('val/Acc_WT', sensitivity_WT, epoch)
        self.writer.add_scalar('val/Acc_TC', sensitivity_TC, epoch)
        self.writer.add_scalar('val/Acc_ET', sensitivity_ET, epoch)
        self.writer.add_scalar('val/mIoU_WT', specificity_WT, epoch)
        self.writer.add_scalar('val/mIoU_TC', specificity_TC, epoch)
        self.writer.add_scalar('val/mIoU_ET', specificity_ET, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, iteration * self.args.batch_size + image.data.shape[0]))
        print("dice：WT:{}, TC:{}, ET:{}, avg: {}"
              .format(dice_WT, dice_TC, dice_ET, dice_avg))
        print("hausdorff：WT:{}, TC:{}, ET:{}, avg: {}"
              .format(hausdorff_WT, hausdorff_TC, hausdorff_ET, hausdorff_avg))
        print("sensitivity: WT:{}, TC:{}, ET:{}, avg: {}"
              .format(sensitivity_WT, sensitivity_TC, sensitivity_ET, sensitivity_avg))
        print("specificity: WT:{}, TC:{}, ET:{}, avg: {}"
              .format(specificity_WT, specificity_TC, specificity_ET, specificity_avg))
        print('Loss: %.3f' % (test_loss / (iteration+1)))

        new_pred = dice_avg
        """"""
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


if __name__ == '__main__':
    args = parse_args()

    print(args)
    trainer = Trainer(args)
    print('Starting Epoch: 0')
    print('Total Epoches:', args.epochs)
    for epoch in range(0, args.epochs):
        trainer.training(epoch)
        trainer.validation(epoch)

    trainer.writer.close()
    torch.cuda.empty_cache()
