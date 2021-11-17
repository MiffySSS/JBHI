import torch
import torch.nn as nn
import torch.nn.functional as f

from backbones import build_backbone
from backbones import build_aspp
from self_sup.encoder_v2 import Encoder


class DecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm_layer=nn.BatchNorm2d):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False),
            norm_layer(out_dim),
            nn.ReLU(True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            norm_layer(out_dim),
            nn.ReLU(True)
        )
    def forward(self, x):
        x = self.block(x)
        return x


class FuseBlock(nn.Module):
    def __init__(self, in_dim, skip_dim, norm_layer=nn.BatchNorm2d):
        super(FuseBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, skip_dim, kernel_size=1, padding=0, bias=False),
            norm_layer(skip_dim),
            nn.ReLU(True)
        )
    def forward(self, x, skip):
        y = f.adaptive_avg_pool2d(x, (1, 1))
        y = self.block(y)
        skip = y * skip
        x = torch.cat((x, skip), dim=1)
        return x


class Decoder(nn.Module):
    def __init__(self, num_classes, norm_layer):
        super(Decoder, self).__init__()
        self.mod1 = DecoderBlock(256, 256, norm_layer)
        self.mod2 = DecoderBlock(320, 256, norm_layer)
        self.mod3 = DecoderBlock(320, 256, norm_layer)
        self.mod4 = DecoderBlock(320, 256, norm_layer)
        self.fuse1 = FuseBlock(256, 64, norm_layer)
        self.fuse2 = FuseBlock(256, 64, norm_layer)
        self.fuse3 = FuseBlock(256, 64, norm_layer)
        self.bn1 = norm_layer(256)
        self.bn2 = norm_layer(256)
        self.bn3 = norm_layer(256)
        self.bn4 = norm_layer(num_classes)
        self.last_conv = nn.Sequential(
            nn.Conv2d(256, 3, 1, padding=0),
            norm_layer(3),
            nn.ReLU(True)
        )
        self._init_weight()

    def forward(self, x, skip1, skip2 ,skip3):
        x = self.mod1(x)
        x = f.interpolate(x, size=skip1.size()[2:], mode='bilinear', align_corners=True)
        x = self.bn1(x)
        x = self.fuse1(x, skip1)
        x = self.mod2(x)
        x = f.interpolate(x, size=skip2.size()[2:], mode='bilinear', align_corners=True)
        x = self.bn2(x)
        x = self.fuse2(x, skip2)
        x = self.mod3(x)
        x = f.interpolate(x, size=skip3.size()[2:], mode='bilinear', align_corners=True)
        x = self.bn3(x)
        x = self.fuse3(x, skip3)
        x = self.mod4(x)
        x = self.last_conv(x)
        x = f.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        x = self.bn4(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class multi_modality_seg(nn.Module):
    def __init__(self, in_plan, num_classes, backbone, norm_layer=nn.BatchNorm2d):
        super(multi_modality_seg, self).__init__()
        self.encoder = Encoder(in_plan=in_plan, backbone=backbone, norm_layer=norm_layer)
        self.decoder = Decoder(num_classes=num_classes, norm_layer=norm_layer)

    def forward(self, x):
        feat, skip1, skip2, skip3 = self.encoder(x)
        output = self.decoder(feat, skip1, skip2, skip3)

        return output


if __name__ == '__main__':
    img = torch.rand(8, 4, 200, 200).cuda()
    model = multi_modality_seg(4, 3, backbone='resnet50').cuda()
    model.eval()
    output = model(img)
    print(output.size())
    for o in output:
        print(o.min(), o.max())
