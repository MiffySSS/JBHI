import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, norm_layer=nn.BatchNorm2d,
                 groups=1, base_width=64):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            print(groups, base_width)
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, dilation, dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, dilation, dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, head_plan, output_stride, norm_layer, pretrained=True, dilated=False, f_lenth=1):
        self.inplanes = int(64 * f_lenth)
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(head_plan, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, int(64 * f_lenth), layers[0], stride=strides[0], dilation=dilations[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, int(128 * f_lenth), layers[1], stride=strides[1], dilation=dilations[1], norm_layer=norm_layer)
        if dilated:
            self.layer3 = self._make_layer(block, int(256 * f_lenth), layers[2], stride=strides[2], dilation=dilations[2], norm_layer=norm_layer)
            self.layer4 = self._make_MG_unit(block, int(512 * f_lenth), blocks=blocks, stride=strides[3], dilation=dilations[3], norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, int(256 * f_lenth), layers[2], stride=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, int(512 * f_lenth), layers[3], stride=2, norm_layer=norm_layer)
        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x_2 = self.relu(x)
        x_4 = self.maxpool(x_2)

        x_4 = self.layer1(x_4)
        x_8 = self.layer2(x_4)
        x_16 = self.layer3(x_8)
        x_32 = self.layer4(x_16)
        return x_2, x_4, x_8, x_16, x_32

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
            else:
                print('unloaded model: ', k)
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


def build_resnet(arch, head_plan, output_stride, norm_layer, pretrained=False, f_lenth=1):
    """Constructs a ResNet model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if arch == 'resnet18':
        model = ResNet(BasicBlock, [2, 2, 2, 2], head_plan, output_stride, norm_layer, pretrained=pretrained, f_lenth=f_lenth)
        config = [(2, int(64*f_lenth)), (4, int(64*f_lenth)), (8, int(128*f_lenth)), (16, int(256*f_lenth)), (32, int(512*f_lenth))]
    elif arch == 'resnet34':
        model = ResNet(BasicBlock, [3, 4, 6, 3], head_plan, output_stride, norm_layer, pretrained=pretrained, f_lenth=f_lenth)
        config = [(2, int(64*f_lenth)), (4, int(64*f_lenth)), (8, int(128*f_lenth)), (16, int(256*f_lenth)), (32, int(512*f_lenth))]
    elif arch == 'resnet50':
        model = ResNet(Bottleneck, [3, 4, 6, 3], head_plan, output_stride, norm_layer, pretrained=pretrained, f_lenth=f_lenth)
        config = [(2, int(64*f_lenth)), (4, int(256*f_lenth)), (8, int(512*f_lenth)), (16, int(1024*f_lenth)), (32, int(2048*f_lenth))]
    elif arch == 'resnet101':
        model = ResNet(Bottleneck, [3, 4, 23, 3], head_plan, output_stride, norm_layer, pretrained=pretrained, f_lenth=f_lenth)
        config = [(2, int(64*f_lenth)), (4, int(256*f_lenth)), (8, int(512*f_lenth)), (16, int(1024*f_lenth)), (32, int(2048*f_lenth))]
    elif arch == 'resnet152':
        model = ResNet(Bottleneck, [3, 8, 36, 3], head_plan, output_stride, norm_layer, pretrained=pretrained, f_lenth=f_lenth)
        config = [(2, int(64*f_lenth)), (4, int(256*f_lenth)), (8, int(512*f_lenth)), (16, int(1024*f_lenth)), (32, int(2048*f_lenth))]
    elif arch == 'resnet50_dilated':
        model = ResNet(Bottleneck, [3, 4, 6, 3], head_plan, output_stride, norm_layer, pretrained=pretrained, dilated=True, f_lenth=f_lenth)
        config = [(2, int(128*f_lenth)), (4, int(256*f_lenth)), (8, int(512*f_lenth)), (8, int(1024*f_lenth)), (8, int(2048*f_lenth))]
    elif arch == 'resnet101_dilated':
        model = ResNet(Bottleneck, [3, 4, 23, 3], head_plan, output_stride, norm_layer, pretrained=pretrained, dilated=True, f_lenth=f_lenth)
        config = [(2, int(128*f_lenth)), (4, int(256*f_lenth)), (8, int(512*f_lenth)), (8, int(1024*f_lenth)), (8, int(2048*f_lenth))]
    elif arch == 'resnet152_dilated':
        model = ResNet(Bottleneck, [3, 8, 36, 3], head_plan, output_stride, norm_layer, pretrained=pretrained, dilated=True, f_lenth=f_lenth)
        config = [(2, int(128*f_lenth)), (4, int(256*f_lenth)), (8, int(512*f_lenth)), (8, int(1024*f_lenth)), (8, int(2048*f_lenth))]
    return model, config


if __name__ == "__main__":
    import torch
    input = torch.rand(8, 4, 200, 200)
    model, _ = build_resnet('resnet50', head_plan=4, norm_layer=nn.BatchNorm2d, pretrained=False, output_stride=16)
    output = model(input)
    for o in output:
        print(o.size())
        print(o.max(), o.min())