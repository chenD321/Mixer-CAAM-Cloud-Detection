# sub-parts of the model

import torch
import torch.nn as nn
import torch.nn.functional as F
from network.gumbel_softmax import GumbelSoftmax2D

class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes]) 
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()
 
    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size)) 
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False) 
        return nn.Sequential(prior, conv)
 
    def forward(self, feats):
        h, w = feats.size(2), feats.size(3) #feats(B C H W)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [int(diffX // 2), int(diffX-diffX//2), int(diffY // 2), int(diffY-diffY//2)], 'constant')
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class ChannelAttentionLayer(nn.Module):
    def __init__(self, C_in, C_out, reduction=16, affine=True, BN=nn.BatchNorm2d):
        super(ChannelAttentionLayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(C_in, max(1, C_in // reduction), 1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(max(1, C_in // reduction), C_out, 1, padding=0, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class CAAM(nn.Module):
    def __init__(self, channel):
        super(CAAM, self).__init__()
        self.depth_revise = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Conv2d(32, 2, 1)
        self.GS = GumbelSoftmax2D(hard=True)

        self.channel = channel
        self.conv0 = nn.Conv2d(channel, channel, 1, padding=0)
        self.conv1 = nn.Conv2d(channel, channel, 1, padding=0)
        self.channel_att = ChannelAttentionLayer(self.channel, self.channel)

    def forward(self, x, gumbel=False):
        n, c, h, w = x.size()
        bins = x
        bins = self.depth_revise(bins)
        gate = self.fc(bins)
        bins = self.GS(gate, gumbel=gumbel) * torch.mean(bins, dim=1, keepdim=True)
        x0 = self.conv0(bins[:, 0, :, :].unsqueeze(1) * x)
        x1 = self.conv1(bins[:, 1, :, :].unsqueeze(1) * x)
        x = (x0 + x1) + x
        x = self.channel_att(x)
        return x


class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)  # depth-wise conv
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)  # depth-wise dilation conv
        self.conv1 = nn.Conv2d(dim, dim, 1)  # 1x1 conv

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class HighMixer(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1, **kwargs, ):
        super().__init__()

        self.cnn_in = cnn_in = dim // 2
        self.pool_in = pool_in = dim // 2

        self.cnn_dim = cnn_dim = cnn_in * 2
        self.pool_dim = pool_dim = pool_in * 2

        self.conv1 = nn.Conv2d(cnn_in, cnn_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj1 = nn.Conv2d(cnn_dim, cnn_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False,
                               groups=cnn_dim)
        self.mid_gelu1 = nn.GELU()

        self.Maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        self.proj2 = nn.Conv2d(pool_in, pool_dim, kernel_size=1, stride=1, padding=0)
        self.mid_gelu2 = nn.GELU()

    def forward(self, x):
        # B, C H, W

        cx = x[:, :self.cnn_in, :, :].contiguous()
        cx = self.conv1(cx)
        cx = self.proj1(cx)
        cx = self.mid_gelu1(cx)

        px = x[:, self.cnn_in:, :, :].contiguous()
        px = self.Maxpool(px)
        px = self.proj2(px)
        px = self.mid_gelu2(px)

        hx = torch.cat((cx, px), dim=1)
        return hx


class LowMixer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)  # depth-wise conv
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3) # depth-wise dilation conv
        self.conv1 = nn.Conv2d(dim, dim, 1)  # 1x1 conv

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class Mixer(nn.Module):
    def __init__(self, dim, proj_drop=0., **kwargs, ):
        super().__init__()
        self.low_dim = low_dim = dim // 2
        self.high_dim = high_dim = dim // 2

        self.high_mixer = HighMixer(high_dim)
        self.low_mixer = LowMixer(low_dim)
        # self.lowconv = nn.Conv2d(low_dim, dim, 1)

        self.conv_fuse = nn.Conv2d(low_dim + high_dim * 2, low_dim + high_dim * 2, kernel_size=3, stride=1, padding=1,
                                   bias=False, groups=low_dim + high_dim * 2)
        self.proj = nn.Conv2d(low_dim + high_dim * 2, dim, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape

        hx = x[:, :self.high_dim, :, :].contiguous()
        hx = self.high_mixer(hx)

        lx = x[:, self.high_dim:, :, :].contiguous()
        lx = self.low_mixer(lx)
        # lx = self.lowconv(lx)

        x = torch.cat((hx, lx), dim=1)
        x = x + self.conv_fuse(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x