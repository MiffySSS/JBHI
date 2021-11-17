import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init

__all__ = ['MobileNetV3', 'mobilenet', 'mobilenet_names']


class Hswish(nn.Module):
    def forward(self, x):
        out = x * f.relu6(x + 3., inplace=True) / 6.
        return out


class Hsigmoid(nn.Module):
    def forward(self, x):
        out = f.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4, norm_layer=nn.BatchNorm2d):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(in_size),
            Hsigmoid())

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    """
    expand + depthwise + pointwise
    """
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride, norm_layer=nn.BatchNorm2d):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule
        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = norm_layer(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(out_size)
        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                norm_layer(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se is not None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV3(nn.Module):
    def __init__(self, head_plane, norm_layer=nn.BatchNorm2d):
        super(MobileNetV3, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(head_plane, 16, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(16),
            Hswish(),
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1, norm_layer=nn.BatchNorm2d)
        )
        self.layer1 = nn.Sequential(
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2, norm_layer=nn.BatchNorm2d),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1, norm_layer=nn.BatchNorm2d),
        )
        self.layer2 = nn.Sequential(
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40, norm_layer=nn.BatchNorm2d), 2, norm_layer=nn.BatchNorm2d),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40, norm_layer=nn.BatchNorm2d), 1, norm_layer=nn.BatchNorm2d),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40, norm_layer=nn.BatchNorm2d), 1, norm_layer=nn.BatchNorm2d),
        )
        self.layer3 = nn.Sequential(
            Block(3, 40, 240, 80, Hswish(), None, 2, norm_layer=nn.BatchNorm2d),
            Block(3, 80, 200, 80, Hswish(), None, 1, norm_layer=nn.BatchNorm2d),
            Block(3, 80, 184, 80, Hswish(), None, 1, norm_layer=nn.BatchNorm2d),
            Block(3, 80, 184, 80, Hswish(), None, 1, norm_layer=nn.BatchNorm2d),
            Block(3, 80, 480, 112, Hswish(), SeModule(112, norm_layer=nn.BatchNorm2d), 1, norm_layer=nn.BatchNorm2d),
            Block(3, 112, 672, 112, Hswish(), SeModule(112, norm_layer=nn.BatchNorm2d), 1, norm_layer=nn.BatchNorm2d),
            Block(5, 112, 672, 160, Hswish(), SeModule(160, norm_layer=nn.BatchNorm2d), 1, norm_layer=nn.BatchNorm2d),
        )
        self.layer4 = nn.Sequential(
            Block(5, 160, 672, 160, Hswish(), SeModule(160, norm_layer=nn.BatchNorm2d), 2, norm_layer=nn.BatchNorm2d),
            Block(5, 160, 960, 160, Hswish(), SeModule(160, norm_layer=nn.BatchNorm2d), 1, norm_layer=nn.BatchNorm2d),
            nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(960),
            Hswish()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, norm_layer):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x_2 = self.layer0(x)
        x_4 = self.layer1(x_2)
        x_8 = self.layer2(x_4)
        x_16 = self.layer3(x_8)
        x_32 = self.layer4(x_16)
        return x_2, x_4, x_8, x_16, x_32


def mobilenet(arch, head_plane=3, **kwargs):
    _model = None
    _config = list()
    if arch == 'mobilenetv3':
        _model = MobileNetV3(head_plane, **kwargs)
        _config = [(2, 16), (4, 24), (8, 40), (16, 160), (32, 960)]
    else:
        raise NotImplementedError
    return _model, _config


mobilenet_names = {'mobilenetv3'}

if __name__ == '__main__':
    import torch
    img = torch.randn(2, 3, 2 * 32, 2 * 32)
    model, _ = mobilenet('mobilenetv3', 3)
    with torch.no_grad():
        output = model(img)
        for o in output:
            print(o.size())