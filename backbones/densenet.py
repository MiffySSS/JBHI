import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

__all__ = ['DenseNet', 'densenet', 'densenet_names']


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output
    return bn_function


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, norm_layer=nn.BatchNorm2d):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', norm_layer(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', norm_layer(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, norm_layer=nn.BatchNorm2d):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                norm_layer=norm_layer
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, norm_layer=nn.BatchNorm2d):
        super(_Transition, self).__init__()
        self.add_module('norm', norm_layer(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    def __init__(self, head_plane, growth_rate=32, block_config=(6, 12, 32, 32),
                 num_init_features=64, bn_size=4, drop_rate=0, norm_layer=nn.BatchNorm2d):
        super(DenseNet, self).__init__()
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(head_plane, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', norm_layer(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                norm_layer=norm_layer
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        # Final batch norm
        self.features.add_module('norm5', norm_layer(num_features))

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, norm_layer):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _transition(x, transition_block):
        for module in transition_block:
            x = module(x)
            if isinstance(module, nn.ReLU):
                skip = x
        return x, skip

    def forward(self, x):
        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)

        x = self.features.pool0(x)
        x = self.features.denseblock1(x)
        x, x_4 = self._transition(x, self.features.transition1)

        x = self.features.denseblock2(x)
        x, x_8 = self._transition(x, self.features.transition2)

        x = self.features.denseblock3(x)
        x, x_16 = self._transition(x, self.features.transition3)

        x = self.features.denseblock4(x)
        x_32 = self.features.norm5(x)

        return x_4, x_8, x_16, x_32


def densenet(arch, head_plane, **kwargs):
    _model = None
    _config = list()
    if arch == 'densenet121':
        kwargs['num_init_features'] = 64
        kwargs['growth_rate'] = 32
        kwargs['block_config'] = (6, 12, 24, 16)
        _model = DenseNet(head_plane, **kwargs)
        _config = [(4, 256), (8, 512), (16, 1024), (32, 1024)]
    elif arch == 'densenet169':
        kwargs['num_init_features'] = 64
        kwargs['growth_rate'] = 32
        kwargs['block_config'] = (6, 12, 32, 32)
        _model = DenseNet(head_plane, **kwargs)
        _config = [(4, 256), (8, 512), (16, 1280), (32, 1664)]
    elif arch == 'densenet201':
        kwargs['num_init_features'] = 64
        kwargs['growth_rate'] = 32
        kwargs['block_config'] = (6, 12, 48, 32)
        _model = DenseNet(head_plane, **kwargs)
        _config = [(4, 256), (8, 512), (16, 1792), (32, 1920)]
    elif arch == 'densenet161':
        kwargs['num_init_features'] = 96
        kwargs['growth_rate'] = 48
        kwargs['block_config'] = (6, 12, 36, 24)
        _model = DenseNet(head_plane, **kwargs)
        _config = [(4, 384), (8, 768), (16, 2112), (32, 2208)]
    else:
        raise NotImplementedError
    return _model, _config


densenet_names = {'densenet121', 'densenet169', 'densenet201', 'densenet161'}


if __name__ == '__main__':
    import torch
    img = torch.randn(2, 3, 2 * 32, 2 * 32)
    model, _ = densenet('densenet121', 3)
    with torch.no_grad():
        output = model(img)
        for o in output:
            print(o.size())
