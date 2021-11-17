import torch
import torch.nn as nn
import torch.nn.functional as f

from backbones import build_backbone
from backbones import build_aspp


class SkipBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm_layer=nn.BatchNorm2d):
        super(SkipBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, 1, padding=0),
            norm_layer(out_dim)
        )
    def forward(self, x):
        x = self.block(x)
        return x

class HAFB(nn.Module):
    def __init__(self, in_dim, modal, norm_layer=nn.BatchNorm2d):
        super(HAFB, self).__init__()
        self.modal = modal
        self.conv1 = nn.Conv2d(in_dim * 3, in_dim, 3, 1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_dim, in_dim * 3, 3, 1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.conv3 = nn.Conv2d(in_dim * 3, in_dim, 1, 1, padding=0)
        self.bn = norm_layer(in_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn1 = norm_layer(in_dim)
        self.bn2 = norm_layer(in_dim)
        self.bn3 = norm_layer(in_dim)

    def forward(self, x):
        len_feat = int(x.shape[1] / self.modal)
        s = torch.split(x, len_feat, dim=1)
        if self.modal == 2:
            x1 = s[0]
            x2 = s[1]
            feat_add = self.bn1(x1 + x2)
            feat_mul = self.bn2(x1 * x2)
            feat_max = self.bn3(torch.max(x1, x2))
        elif self.modal == 4:
            x1 = s[0]
            x2 = s[1]
            x3 = s[2]
            x4 = s[3]
            feat_add = self.bn1(x1 + x2 + x3 + x4)
            feat_mul = self.bn2(x1 * x2 * x3 * x4)
            feat_max = self.bn3(torch.max(torch.max(x1, x2), torch.max(x3, x4)))
        else:
            print("input channel error")
        x_a = torch.cat((feat_add, feat_mul, feat_max), dim=1)
        x_b = self.conv1(x_a)
        x_b = self.relu1(x_b)
        x_b = self.conv2(x_b)
        x_b = self. sigmoid(x_b)
        y = x_a * x_b + x_a
        y = self.conv3(y)
        y = self.bn(y)
        y = self.relu2(y)
        return y


class Encoder(nn.Module):
    def __init__(self, in_plan, backbone, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()
        self.in_plan = in_plan
        self.resnet1, self.info_modality1 = build_backbone(backbone=backbone, head_plan=1, output_stride=16, norm_layer=norm_layer)
        self.resnet2, self.info_modality2 = build_backbone(backbone=backbone, head_plan=1, output_stride=16, norm_layer=norm_layer)
        if in_plan == 4:
            self.resnet3, self.info_modality3 = build_backbone(backbone=backbone, head_plan=1, output_stride=16, norm_layer=norm_layer)
            self.resnet4, self.info_modality4 = build_backbone(backbone=backbone, head_plan=1, output_stride=16, norm_layer=norm_layer)
        self.aspp1 = build_aspp(backbone=backbone, output_stride=16, norm_layer=norm_layer)
        self.aspp2 = build_aspp(backbone=backbone, output_stride=16, norm_layer=norm_layer)
        if in_plan == 4:
            self.aspp3 = build_aspp(backbone=backbone, output_stride=16, norm_layer=norm_layer)
            self.aspp4 = build_aspp(backbone=backbone, output_stride=16, norm_layer=norm_layer)
        self.skip1_1 = SkipBlock(256, 64)
        self.skip2_1 = SkipBlock(256, 64)
        if in_plan == 4:
            self.skip3_1 = SkipBlock(256, 64)
            self.skip4_1 = SkipBlock(256, 64)
        self.skip1_2 = SkipBlock(512, 64)
        self.skip2_2 = SkipBlock(512, 64)
        if in_plan == 4:
            self.skip3_2 = SkipBlock(512, 64)
            self.skip4_2 = SkipBlock(512, 64)
        self.skip1_3 = SkipBlock(1024, 64)
        self.skip2_3 = SkipBlock(1024, 64)
        if in_plan == 4:
            self.skip3_3 = SkipBlock(1024, 64)
            self.skip4_3 = SkipBlock(1024, 64)
        self.fusion1 = HAFB(64, in_plan, norm_layer)
        self.fusion2 = HAFB(64, in_plan, norm_layer)
        self.fusion3 = HAFB(64, in_plan, norm_layer)
        self.fusion4 = HAFB(256, in_plan, norm_layer)

    def forward(self, x):
        m1_2, m1_4, m1_8, m1_16, m1_32 = self.resnet1(x[:, 0, None, :, :])
        m1_4 = self.skip1_1(m1_4)
        m1_8 = self.skip1_2(m1_8)
        m1_16 = self.skip1_3(m1_16)
        feat1 = self.aspp1(m1_32)
        m2_2, m2_4, m2_8, m2_16, m2_32 = self.resnet2(x[:, 1, None, :, :])
        m2_4 = self.skip2_1(m2_4)
        m2_8 = self.skip2_2(m2_8)
        m2_16 = self.skip2_3(m2_16)
        feat2 = self.aspp2(m2_32)
        if self.in_plan == 4:
            m3_2, m3_4, m3_8, m3_16, m3_32 = self.resnet3(x[:, 2, None, :, :])
            m3_4 = self.skip3_1(m3_4)
            m3_8 = self.skip3_2(m3_8)
            m3_16 = self.skip3_3(m3_16)
            feat3 = self.aspp3(m3_32)
            m4_2, m4_4, m4_8, m4_16, m4_32 = self.resnet4(x[:, 3, None, :, :])
            m4_4 = self.skip4_1(m4_4)
            m4_8 = self.skip4_2(m4_8)
            m4_16 = self.skip4_3(m4_16)
            feat4 = self.aspp4(m4_32)
        if self.in_plan == 2:
            m16 = torch.cat((m1_16, m2_16), dim=1)
            m8 = torch.cat((m1_8, m2_8), dim=1)
            m4 = torch.cat((m1_4, m2_4), dim=1)
            feat = torch.cat((feat1, feat2), dim=1)
        elif self.in_plan == 4:
            m16 = torch.cat((m1_16, m2_16, m3_16, m4_16), dim=1)
            m8 = torch.cat((m1_8, m2_8, m3_8, m4_8), dim=1)
            m4 = torch.cat((m1_4, m2_4, m3_4, m4_4), dim=1)
            feat = torch.cat((feat1, feat2, feat3, feat4), dim=1)
        else:
            print("input channel error")
        fuse_skip1 = self.fusion1(m16)
        fuse_skip2 = self.fusion2(m8)
        fuse_skip3 = self.fusion3(m4)
        fuse_feat = self.fusion4(feat)

        return fuse_feat, fuse_skip1, fuse_skip2, fuse_skip3


if __name__ == '__main__':
    img = torch.rand(8, 4, 200, 200).cuda()
    model = Encoder(4, backbone='resnet50').cuda()
    model.eval()
    output = model(img)
    for o in output:
        print(o.size())
        print(o.min(), o.max())
