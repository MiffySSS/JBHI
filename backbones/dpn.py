import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

__all__ = ['DPN', 'dpn', 'dpn_names']


class CatBnAct(nn.Module):
    def __init__(self, in_chs, activation_fn=nn.ReLU(inplace=True), norm_layer=nn.BatchNorm2d):
        super(CatBnAct, self).__init__()
        self.bn = norm_layer(in_chs, eps=0.001)
        self.act = activation_fn

    def forward(self, x):
        x = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        return self.act(self.bn(x))


class BnActConv2d(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride, padding=0, groups=1,
                 activation_fn=nn.ReLU(inplace=True), norm_layer=nn.BatchNorm2d):
        super(BnActConv2d, self).__init__()
        self.bn = norm_layer(in_chs, eps=0.001)
        self.act = activation_fn
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, groups=groups, bias=False)

    def forward(self, x):
        return self.conv(self.act(self.bn(x)))


class InputBlock(nn.Module):
    def __init__(self, head_plane, num_init_features, kernel_size=7, padding=3, norm_layer=nn.BatchNorm2d):
        super(InputBlock, self).__init__()
        self.conv = nn.Conv2d(head_plane, num_init_features, kernel_size=kernel_size, stride=2, padding=padding, bias=False)
        self.bn = norm_layer(num_init_features, eps=0.001)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class DualPathBlock(nn.Module):
    def __init__(self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, groups, block_type='normal', norm_layer=nn.BatchNorm2d):
        super(DualPathBlock, self).__init__()
        self.num_1x1_c = num_1x1_c
        self.inc = inc
        if block_type == 'proj':
            self.key_stride = 1
            self.has_proj = True
        elif block_type == 'down':
            self.key_stride = 2
            self.has_proj = True
        else:
            assert block_type == 'normal'
            self.key_stride = 1
            self.has_proj = False

        if self.has_proj:
            # Using different member names here to allow easier parameter key matching for conversion
            if self.key_stride == 2:
                self.c1x1_w_s2 = BnActConv2d(in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=2, norm_layer=norm_layer)
            else:
                self.c1x1_w_s1 = BnActConv2d(in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=1, norm_layer=norm_layer)
        self.c1x1_a = BnActConv2d(in_chs=in_chs, out_chs=num_1x1_a, kernel_size=1, stride=1, norm_layer=norm_layer)
        self.c3x3_b = BnActConv2d(in_chs=num_1x1_a, out_chs=num_3x3_b, kernel_size=3, stride=self.key_stride, padding=1, groups=groups, norm_layer=norm_layer)
        self.c1x1_c = BnActConv2d(in_chs=num_3x3_b, out_chs=num_1x1_c + inc, kernel_size=1, stride=1, norm_layer=norm_layer)

    def forward(self, x):
        x_in = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        if self.has_proj:
            x_s = self.c1x1_w_s2(x_in) if self.key_stride == 2 else self.c1x1_w_s1(x_in)
            x_s1 = x_s[:, :self.num_1x1_c, :, :]
            x_s2 = x_s[:, self.num_1x1_c:, :, :]
        else:
            x_s1 = x[0]
            x_s2 = x[1]
        x_in = self.c1x1_a(x_in)
        x_in = self.c3x3_b(x_in)

        x_in = self.c1x1_c(x_in)
        out1 = x_in[:, :self.num_1x1_c, :, :]
        out2 = x_in[:, self.num_1x1_c:, :, :]
        resid = x_s1 + out1
        dense = torch.cat([x_s2, out2], dim=1)
        return resid, dense


class DPN(nn.Module):
    def __init__(self, head_plane, num_init_features=64, k_r=96, groups=32,
                 k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128), norm_layer=nn.BatchNorm2d):
        super(DPN, self).__init__()
        bw_factor = 4
        blocks = OrderedDict()

        # conv1
        blocks['conv1_1'] = InputBlock(head_plane, num_init_features, kernel_size=7, padding=3, norm_layer=norm_layer)

        # conv2
        bw = 64 * bw_factor
        inc = inc_sec[0]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv2_1'] = DualPathBlock(num_init_features, r, r, bw, inc, groups, 'proj', norm_layer=norm_layer)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[0] + 1):
            blocks['conv2_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', norm_layer=norm_layer)
            in_chs += inc

        # conv3
        bw = 128 * bw_factor
        inc = inc_sec[1]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv3_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', norm_layer=norm_layer)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[1] + 1):
            blocks['conv3_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', norm_layer=norm_layer)
            in_chs += inc

        # conv4
        bw = 256 * bw_factor
        inc = inc_sec[2]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv4_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', norm_layer=norm_layer)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[2] + 1):
            blocks['conv4_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', norm_layer=norm_layer)
            in_chs += inc

        # conv5
        bw = 512 * bw_factor
        inc = inc_sec[3]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv5_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', norm_layer=norm_layer)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[3] + 1):
            blocks['conv5_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', norm_layer=norm_layer)
            in_chs += inc
        blocks['conv5_bn_ac'] = CatBnAct(in_chs, norm_layer=norm_layer)

        self.features = nn.Sequential(blocks)
        self.feature_blocks = np.cumsum(k_sec)
        self.feature_blocks[-1] += 1

    def forward(self, x):
        features = list()
        input_block = self.features[0]
        x = input_block.conv(x)
        x = input_block.bn(x)
        x = input_block.act(x)
        features.append(x)

        x = input_block.pool(x)

        for i, module in enumerate(self.features[1:], 1):
            x = module(x)
            if i in self.feature_blocks:
                features.append(x)

        x_4 = F.relu(torch.cat(features[1], dim=1), inplace=True)
        x_8 = F.relu(torch.cat(features[2], dim=1), inplace=True)
        x_16 = F.relu(torch.cat(features[3], dim=1), inplace=True)
        x_32 = features[4]

        return x_4, x_8, x_16, x_32


def dpn(arch, head_plane, **kwargs):
    _model = None
    _config = list()
    if arch == 'dpn92':
        kwargs['num_init_features'] = 64
        kwargs['groups'] = 32
        kwargs['inc_sec'] = (16, 32, 24, 128)
        kwargs['k_r'] = 96
        kwargs['k_sec'] = (3, 4, 20, 3)
        _model = DPN(head_plane, **kwargs)
        _config = [(4, 336), (8, 704), (16, 1552), (32, 2688)]
    elif arch == 'dpn98':
        kwargs['num_init_features'] = 96
        kwargs['groups'] = 40
        kwargs['inc_sec'] = (16, 32, 32, 128)
        kwargs['k_r'] = 160
        kwargs['k_sec'] = (3, 6, 20, 3)
        _model = DPN(head_plane, **kwargs)
        _config = [(4, 336), (8, 768), (16, 1728), (32, 2688)]
    elif arch == 'dpn107':
        kwargs['num_init_features'] = 128
        kwargs['groups'] = 50
        kwargs['inc_sec'] = (20, 64, 64, 128)
        kwargs['k_r'] = 200
        kwargs['k_sec'] = (4, 8, 20, 3)
        _model = DPN(head_plane, **kwargs)
        _config = [(4, 376), (8, 1152), (16, 2432), (32, 2688)]
    elif arch == 'dpn131':
        kwargs['num_init_features'] = 128
        kwargs['groups'] = 40
        kwargs['inc_sec'] = (16, 32, 32, 128)
        kwargs['k_r'] = 160
        kwargs['k_sec'] = (4, 8, 28, 3)
        _model = DPN(head_plane, **kwargs)
        _config = [(4, 352), (8, 832), (16, 1984), (32, 2688)]
    else:
        raise NotImplementedError
    return _model, _config


dpn_names = {'dpn92', 'dpn98', 'dpn107', 'dpn131'}


if __name__ == '__main__':
    import torch
    img = torch.randn(2, 3, 2 * 32, 2 * 32)
    model, _ = dpn('dpn92', 3)
    with torch.no_grad():
        output = model(img)
        for o in output:
            print(o.size())



