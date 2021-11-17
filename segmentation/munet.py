import torch
import torch.nn as nn
import torch.nn.functional as f

from backbones import build_backbone


class DecoderBlock(nn.Module):
    def __init__(self, x_dim, skip_dim, out_dim, norm_layer=nn.BatchNorm2d):
        super(DecoderBlock, self).__init__()

        self.block = torch.nn.Sequential(
            nn.Conv2d(x_dim + skip_dim, out_dim, kernel_size=3, padding=1, bias=False),
            norm_layer(out_dim),
            nn.ReLU(True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            norm_layer(out_dim),
            nn.ReLU(True)
        )

    def forward(self, x, skip):
        # upsample
        x = f.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=True)
        # fusion
        return self.block(torch.cat((x, skip), dim=1))


class UNetHead(nn.Module):
    def __init__(self, nclass, enc_dims, norm_layer=nn.BatchNorm2d):
        super(UNetHead, self).__init__()
        self.enc_dims = enc_dims[::-1]
        self.out_dims = [256, 128, 64, 32, 16]
        self.norm_lyaer = norm_layer
        self.up_c5 = DecoderBlock(self.enc_dims[0], self.enc_dims[1], self.out_dims[0])
        self.up_c4 = DecoderBlock(self.out_dims[0], self.enc_dims[2], self.out_dims[1])
        self.up_c3 = DecoderBlock(self.out_dims[1], self.enc_dims[3], self.out_dims[2])
        self.up_c2 = DecoderBlock(self.out_dims[2], self.enc_dims[4], self.out_dims[3])
        self.up_c1 = nn.Sequential(
            nn.Conv2d(self.out_dims[3], self.out_dims[4], kernel_size=3, padding=1, bias=False),
            norm_layer(self.out_dims[4]),
            nn.ReLU(True),
            nn.Conv2d(self.out_dims[4], nclass, kernel_size=1)
        )

    def forward(self, c1, c2, c3, c4, c5):
        f4 = self.up_c5(c5, c4)
        f3 = self.up_c4(f4, c3)
        f2 = self.up_c3(f3, c2)
        f1 = self.up_c2(f2, c1)
        f1 = torch.nn.functional.interpolate(f1, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.up_c1(f1)
        return out


class MUNet(torch.nn.Module):
    def __init__(self, head_plane, num_class, backbone, norm_layer=nn.BatchNorm2d):
        super(MUNet, self).__init__()
        self.head_plane = head_plane
        self.num_class = num_class
        self.feats1, self.info1 = build_backbone(backbone=backbone, head_plan=1, output_stride=16, norm_layer=norm_layer)
        self.feats2, self.info2 = build_backbone(backbone=backbone, head_plan=1, output_stride=16, norm_layer=norm_layer)
        self.feats3, self.info3 = build_backbone(backbone=backbone, head_plan=1, output_stride=16, norm_layer=norm_layer)
        self.feats4, self.info4 = build_backbone(backbone=backbone, head_plan=1, output_stride=16, norm_layer=norm_layer)

        enc_dims = [cfg[1] for cfg in self.info1]
        self.conv1 = nn.Conv2d(enc_dims[0] * head_plane, enc_dims[0], 1, 1, padding=0)
        self.conv2 = nn.Conv2d(enc_dims[1] * head_plane, enc_dims[1], 1, 1, padding=0)
        self.conv3 = nn.Conv2d(enc_dims[2] * head_plane, enc_dims[2], 1, 1, padding=0)
        self.conv4 = nn.Conv2d(enc_dims[3] * head_plane, enc_dims[3], 1, 1, padding=0)
        self.conv5 = nn.Conv2d(enc_dims[4] * head_plane, enc_dims[4], 1, 1, padding=0)

        self.head = UNetHead(num_class, enc_dims, norm_layer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):

        c1_1, c2_1, c3_1, c4_1, c5_1 = self.feats1.forward(x[:, 0, None, :, :])
        if self.head_plane == 2:
            c1_2, c2_2, c3_2, c4_2, c5_2 = self.feats2.forward(x[:, 1, None, :, :])
        if self.head_plane == 4:
            c1_2, c2_2, c3_2, c4_2, c5_2 = self.feats2.forward(x[:, 1, None, :, :])
            c1_3, c2_3, c3_3, c4_3, c5_3 = self.feats3.forward(x[:, 2, None, :, :])
            c1_4, c2_4, c3_4, c4_4, c5_4 = self.feats4.forward(x[:, 3, None, :, :])

        if self.head_plane == 1:
            c1 = c1_1
            c2 = c2_1
            c3 = c3_1
            c4 = c4_1
            c5 = c5_1
        elif self.head_plane == 2:
            c1 = torch.cat((c1_1, c1_2), dim=1)
            c2 = torch.cat((c2_1, c2_2), dim=1)
            c3 = torch.cat((c3_1, c3_2), dim=1)
            c4 = torch.cat((c4_1, c4_2), dim=1)
            c5 = torch.cat((c5_1, c5_2), dim=1)
        elif self.head_plane == 4:
            c1 = torch.cat((c1_1, c1_2, c1_3, c1_4), dim=1)
            c2 = torch.cat((c2_1, c2_2, c2_3, c2_4), dim=1)
            c3 = torch.cat((c3_1, c3_2, c3_3, c3_4), dim=1)
            c4 = torch.cat((c4_1, c4_2, c4_3, c4_4), dim=1)
            c5 = torch.cat((c5_1, c5_2, c5_3, c5_4), dim=1)
        else:
            print('input channel error')

        c1 = self.conv1(c1)
        c2 = self.conv2(c2)
        c3 = self.conv3(c3)
        c4 = self.conv4(c4)
        c5 = self.conv5(c5)
        logit = self.head(c1, c2, c3, c4, c5)

        return logit

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.feats]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.head]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


if __name__ == '__main__':

    img = torch.randn(2, 4, 200, 200).cuda()
    gt = torch.randn(2, 3, 200, 200).cuda()
    model = MUNet(2, 3, backbone='resnet50').cuda()
    #model.training = False
    with torch.no_grad():
        output = model(img)
        for o in output:
            print(o.shape)
            print(o.min(), o.max())

