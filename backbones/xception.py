import math
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Xception', 'xception', 'xception_names']


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False, norm_layer=None):
        super(SeparableConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 0, dilation, groups=in_channels, bias=bias)
        self.bn = norm_layer(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.fix_padding(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

    def fix_padding(self, x):
        kernel_size_effective = self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        padded_inputs = F.pad(x, [pad_beg, pad_end, pad_beg, pad_end])
        return padded_inputs


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, reps, stride=1, dilation=1, norm_layer=None, start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()
        if out_channels != in_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
            self.skipbn = norm_layer(out_channels)
        else:
            self.skip = None
        rep = list()
        filters = in_channels
        if grow_first:
            if start_with_relu:
                rep.append(nn.ReLU(True))
            rep.append(SeparableConv2d(in_channels, out_channels, 3, 1, dilation, norm_layer=norm_layer))
            rep.append(norm_layer(out_channels))
            filters = out_channels
        for i in range(reps - 1):
            if grow_first or start_with_relu:
                rep.append(nn.ReLU(True))
            rep.append(SeparableConv2d(filters, filters, 3, 1, dilation, norm_layer=norm_layer))
            rep.append(norm_layer(filters))
        if not grow_first:
            rep.append(nn.ReLU(True))
            rep.append(SeparableConv2d(in_channels, out_channels, 3, 1, dilation, norm_layer=norm_layer))
        if stride != 1:
            rep.append(nn.ReLU(True))
            rep.append(SeparableConv2d(out_channels, out_channels, 3, stride, norm_layer=norm_layer))
            rep.append(norm_layer(out_channels))
        elif is_last:
            rep.append(nn.ReLU(True))
            rep.append(SeparableConv2d(out_channels, out_channels, 3, 1, dilation, norm_layer=norm_layer))
            rep.append(norm_layer(out_channels))
        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        out = self.rep(x)
        if self.skip is not None:
            skip = self.skipbn(self.skip(x))
        else:
            skip = x
        out = out + skip
        return out


class Xception(nn.Module):
    def __init__(self, head_plane, output_stride=8, norm_layer=nn.BatchNorm2d):
        super(Xception, self).__init__()
        if output_stride == 32:
            entry_block3_stride = 2
            exit_block20_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 1)
        elif output_stride == 16:
            entry_block3_stride = 2
            exit_block20_stride = 1
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            exit_block20_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError
        # Entry flow
        self.conv1 = nn.Conv2d(head_plane, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(32)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(64)

        self.block1 = Block(64, 128, reps=2, stride=2, norm_layer=norm_layer, start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, norm_layer=norm_layer, start_with_relu=False, grow_first=True)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride, norm_layer=norm_layer,
                            start_with_relu=True, grow_first=True, is_last=True)

        # Middle flow
        midflow = list()
        for i in range(4, 20):
            midflow.append(Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, norm_layer=norm_layer,
                                 start_with_relu=True, grow_first=True))
        self.midflow = nn.Sequential(*midflow)

        # Exit flow
        self.block20 = Block(728, 1024, reps=2, stride=exit_block20_stride, dilation=exit_block_dilations[0],
                             norm_layer=norm_layer, start_with_relu=True, grow_first=False, is_last=True)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, dilation=exit_block_dilations[1], norm_layer=norm_layer)
        self.bn3 = norm_layer(1536)
        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=exit_block_dilations[1], norm_layer=norm_layer)
        self.bn4 = norm_layer(1536)
        self.conv5 = SeparableConv2d(1536, 2048, 3, 1, dilation=exit_block_dilations[1], norm_layer=norm_layer)
        self.bn5 = norm_layer(2048)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # Entry flow
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        x = self.block1(x)
        x = self.relu(x)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)
        x = self.midflow(x)
        mid_level_feat = x

        # Exit flow
        x = self.block20(x)
        x = self.bn3(self.conv3(self.relu(x)))
        x = self.bn4(self.conv4(self.relu(x)))
        x = self.bn5(self.conv5(self.relu(x)))
        x = self.relu(x)
        high_level_feat = x

        return low_level_feat, mid_level_feat, high_level_feat


def xception(arch, head_plane, **kwargs):
    _model = None
    _config = list()
    if arch == 'xception_8':
        _model = Xception(head_plane, 8, **kwargs)
        _config = [(4, 128), (8, 728), (8, 2048)]
    elif arch == 'xception_16':
        _model = Xception(head_plane, 16, **kwargs)
        _config = [(4, 128), (16, 728), (16, 2048)]
    elif arch == 'xception_32':
        _model = Xception(head_plane, 32, **kwargs)
        _config = [(4, 128), (16, 728), (32, 2048)]
    else:
        raise NotImplementedError

    return _model, _config


xception_names = {'xception_8', 'xception_16', 'xception_32'}


if __name__ == '__main__':
    import torch
    img = torch.randn(2, 3, 2 * 32, 2 * 32)
    model, _ = xception('xception_8', 3)
    with torch.no_grad():
        output = model(img)
        for o in output:
            print(o.size())