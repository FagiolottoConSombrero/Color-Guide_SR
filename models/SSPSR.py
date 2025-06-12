import torch
import math
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    if dilation==1:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=(kernel_size//2), bias=bias)
    elif dilation==2:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=2, bias=bias, dilation=dilation)

    else:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=3, bias=bias, dilation=dilation)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class ResAttentionBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResAttentionBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        m.append(CALayer(n_feats, 16))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class SSB(nn.Module):
    def __init__(self, n_feats, kernel_size, act, res_scale, conv=default_conv):
        super(SSB, self).__init__()
        self.spa = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.spc = ResAttentionBlock(conv, n_feats, 1, act=act, res_scale=res_scale)

    def forward(self, x):
        return self.spc(self.spa(x))


class SSPN(nn.Module):
    def __init__(self, n_feats, n_blocks, act, res_scale):
        super(SSPN, self).__init__()

        kernel_size = 3
        m = []

        for i in range(n_blocks):
            m.append(SSB(n_feats, kernel_size, act=act, res_scale=res_scale))

        self.net = nn.Sequential(*m)

    def forward(self, x):
        res = self.net(x)
        res += x

        return res


# a single branch of proposed SSPSR
class BranchUnit(nn.Module):
    def __init__(self, n_colors, n_feats, n_blocks, act, res_scale, up_scale, use_tail=True, conv=default_conv):
        super(BranchUnit, self).__init__()
        kernel_size = 3
        self.head = conv(n_colors, n_feats, kernel_size)
        self.body = SSPN(n_feats, n_blocks, act, res_scale)
        self.upsample = Upsampler(conv, up_scale, n_feats)
        self.tail = None

        if use_tail:
            self.tail = conv(n_feats, n_colors, kernel_size)

    def forward(self, x):
        y = self.head(x)
        y = self.body(y)
        y = self.upsample(y)
        if self.tail is not None:
            y = self.tail(y)

        return y


class SSPSR(nn.Module):
    def __init__(self, n_subs, n_ovls, n_colors, n_blocks, n_feats, n_scale, res_scale, use_share=True, conv=default_conv):
        super(SSPSR, self).__init__()
        kernel_size = 3
        self.shared = use_share
        act = nn.ReLU(True)

        # calculate the group number (the number of branch networks)
        self.G = math.ceil((n_colors - n_ovls) / (n_subs - n_ovls))
        # calculate group indices
        self.start_idx = []
        self.end_idx = []

        for g in range(self.G):
            sta_ind = (n_subs - n_ovls) * g
            end_ind = sta_ind + n_subs
            if end_ind > n_colors:
                end_ind = n_colors
                sta_ind = n_colors - n_subs
            self.start_idx.append(sta_ind)
            self.end_idx.append(end_ind)

        if self.shared:
            self.branch = BranchUnit(n_subs, n_feats, n_blocks, act, res_scale, up_scale=n_scale//2, conv=default_conv)
            # up_scale=n_scale//2 means that we upsample the LR input n_scale//2 at the branch network, and then conduct 2 times upsampleing at the global network
        else:
            self.branch = nn.ModuleList()
            for i in range(self.G):
                self.branch.append(BranchUnit(n_subs, n_feats, n_blocks, act, res_scale, up_scale=2, conv=default_conv))

        self.trunk = BranchUnit(n_colors, n_feats, n_blocks, act, res_scale, up_scale=2, use_tail=False, conv=default_conv)
        self.skip_conv = conv(3, n_feats, kernel_size)
        self.final = conv(n_feats, n_colors, kernel_size)
        self.sca = n_scale//2

    def forward(self, x, lms):
        b, c, h, w = x.shape

        # Initialize intermediate “result”, which is upsampled with n_scale//2 times
        y = torch.zeros(b, c, self.sca * h, self.sca * w).cuda()
        # y = torch.zeros(b, c, self.sca * h, self.sca * w)

        channel_counter = torch.zeros(c).cuda()
        # channel_counter = torch.zeros(c)

        for g in range(self.G):
            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]

            xi = x[:, sta_ind:end_ind, :, :]
            if self.shared:
                xi = self.branch(xi)
            else:
                xi = self.branch[g](xi)

            y[:, sta_ind:end_ind, :, :] += xi
            channel_counter[sta_ind:end_ind] = channel_counter[sta_ind:end_ind] + 1

        # intermediate “result” is averaged according to their spectral indices
        y = y / channel_counter.unsqueeze(1).unsqueeze(2)

        y = self.trunk(y)

        y = y + self.skip_conv(lms)
        y = self.final(y)

        return y