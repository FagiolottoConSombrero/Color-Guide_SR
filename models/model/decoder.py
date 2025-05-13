from models.model.common import *
import torch


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.first_conv = self.pixelshuffle_block(128, 64)
        self.second_conv = self.pixelshuffle_block(192, 64)
        self.last_conv = Conv3XC(192, 31, gain1=1, s=1, relu=True)

    def pixelshuffle_block(self,
                           in_channels,
                           out_channels,
                           upscale_factor=2,
                           kernel_size=5):
        conv = nn.Conv2d(in_channels,
                         out_channels * (upscale_factor ** 2),
                         kernel_size=kernel_size,
                         padding=kernel_size // 2)
        pixel_shuffle = nn.PixelShuffle(upscale_factor)
        return nn.Sequential(conv, pixel_shuffle, nn.LeakyReLU(0.1, inplace=True))

    def make_conv(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

    def forward(self, hsi_feats, rgb_feats):
        x = torch.cat((hsi_feats[0], rgb_feats[2]), dim=1)
        x = self.first_conv(x)

        conc = torch.cat((x, hsi_feats[1], rgb_feats[1]), dim=1)
        x = self.second_conv(conc)

        conc = torch.cat((x, hsi_feats[2], rgb_feats[0]), dim=1)
        x = self.last_conv(conc)
        return x
