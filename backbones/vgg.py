import torch.nn as nn

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, cfg, head_plane, batch_norm=False):
        super(VGG, self).__init__()
        self.head_plane = head_plane
        self.batch_norm = batch_norm
        self.group1, self.group2, self.group3, self.group4, self.group5 = self.make_layers(cfg)

    def make_layers(self, cfg):
        layers = list()
        groups = list()
        in_channels = self.head_plane
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                groups.append(nn.Sequential(*layers))
                layers = list()
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if self.batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return tuple(groups)

    def forward(self, x):
        x_2 = self.group1(x)
        x_4 = self.group2(x_2)
        x_8 = self.group3(x_4)
        x_16 = self.group4(x_8)
        x_32 = self.group5(x_16)

        return x_2, x_4, x_8, x_16, x_32


def vgg(arch, head_plane=3, **kwargs):
    _model = None
    _config = list()
    if arch == 'vgg11':
        _model = VGG(cfgs['A'], head_plane, **kwargs)
        _config = [(2, 64), (4, 128), (8, 256), (16, 512), (32, 512)]
    elif arch == 'vgg11_bn':
        kwargs['batch_norm'] = True
        _model = VGG(cfgs['A'], head_plane, **kwargs)
        _config = [(2, 64), (4, 128), (8, 256), (16, 512), (32, 512)]
    elif arch == 'vgg13':
        _model = VGG(cfgs['B'], head_plane, **kwargs)
        _config = [(2, 64), (4, 128), (8, 256), (16, 512), (32, 512)]
    elif arch == 'vgg13_bn':
        kwargs['batch_norm'] = True
        _model = VGG(cfgs['B'], head_plane, **kwargs)
        _config = [(2, 64), (4, 128), (8, 256), (16, 512), (32, 512)]
    elif arch == 'vgg16':
        _model = VGG(cfgs['D'], head_plane, **kwargs)
        _config = [(2, 64), (4, 128), (8, 256), (16, 512), (32, 512)]
    elif arch == 'vgg16_bn':
        kwargs['batch_norm'] = True
        _model = VGG(cfgs['D'], head_plane, **kwargs)
        _config = [(2, 64), (4, 128), (8, 256), (16, 512), (32, 512)]
    elif arch == 'vgg19':
        _model = VGG(cfgs['E'], head_plane, **kwargs)
        _config = [(2, 64), (4, 128), (8, 256), (16, 512), (32, 512)]
    elif arch == 'vgg19_bn':
        kwargs['batch_norm'] = True
        _model = VGG(cfgs['E'], head_plane, **kwargs)
        _config = [(2, 64), (4, 128), (8, 256), (16, 512), (32, 512)]
    else:
        raise NotImplementedError
    return _model, _config


vgg_names = {'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'}

if __name__ == '__main__':
    import torch
    img = torch.randn(2, 3, 2 * 32, 2 * 32)
    model, _ = vgg('vgg19_bn', 3)
    with torch.no_grad():
        output = model(img)
        for o in output:
            print(o.size())