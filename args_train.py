import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    # model and dataset
    parser.add_argument('--modalities', type=str, default='all',
                        choices=['all', 'flair', 't1', 't2', 't1ce', 't1ce+flair', 't1ce+t2', 't1ce+t1'],
                        help='modalities (default: all)')
    parser.add_argument('--model', type=str, default='unet',
                        choices=['unet', 'deeplab', 'multi_modal_seg', 'munet', 'transformer'],
                        help='model name (default: unet)')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet', 'vgg16', 'resnet18', 'resnet50', 'resnet101', 'resnet152',
                                 'resnet50_dilated', 'resnet101_dilated', 'resnet152_dilated'],
                        help='backbone name (default: resnet18)')
    parser.add_argument('--out-stride', type=int, default=1,
                        help='network output stride (default: 1)')
    parser.add_argument('--dataset', type=str, default='brats',
                        choices=['brats'],
                        help='dataset name (default: brats)')
    parser.add_argument('--workers', '-j', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=240,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=200,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='dice',
                        choices=['bce', 'ce', 'bfocal', 'focal', 'dice'],
                        help='loss func type (default: dice)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                testing (default: 1)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 1e-2)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0, metavar='M',
                        help='w-decay (default: 0)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='3',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=3)')
    
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default='./results',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log-dir', default='./runs/logs/',
                        help='Directory for saving training logs')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation only
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()

    return args