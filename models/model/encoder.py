from models.model.common import *
import torch


class RGBEncoder(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, stride=1):
        super(RGBEncoder, self).__init__()

        self.initial_layer = Conv3XC(in_ch, out_ch, gain1=1, s=stride, relu=True, padding=2)
        self.test_layer = Conv3XC(out_ch, out_ch, gain1=1, s=stride, relu=True)
        self.test_layer_2 = Conv3XC(out_ch, out_ch, gain1=1, s=2, relu=True)

    def forward(self, x):
        x = self.initial_layer(x)
        conv1 = self.test_layer(x)
        conv2 = self.test_layer_2(conv1)
        conv3 = self.test_layer_2(conv2)
        return [conv1, conv2, conv3]


class RGBEncoder6(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, stride=1):
        super(RGBEncoder6, self).__init__()

        self.initial_layer = Conv3XC(in_ch, out_ch, gain1=1, s=stride, relu=True, padding=2)
        self.test_layer = Conv3XC(out_ch, out_ch, gain1=1, s=stride, relu=True)
        self.test_layer_2 = Conv3XC(out_ch, out_ch, gain1=1, s=3, relu=True)
        self.test_layer_3 = Conv3XC(out_ch, out_ch, gain1=1, s=2, relu=True)

    def forward(self, x):
        x = self.initial_layer(x)
        conv1 = self.test_layer(x)
        conv2 = self.test_layer_2(conv1)
        conv3 = self.test_layer_3(conv2)
        return [conv1, conv2, conv3]


class HSIEncoder(nn.Module):
    def __init__(self, in_ch=31, out_ch=64, kernel_size=3):
        super(HSIEncoder, self).__init__()
        self.first_layer = Conv3XC(in_ch, out_ch, gain1=1, s=1, relu=True)
        self.pixel_upsample = self.pixelshuffle_block(out_ch, out_ch)
        self.pixel_upsample_2 = self.pixelshuffle_block(out_ch, out_ch)

    def pixelshuffle_block(self,
                           in_channels,
                           out_channels,
                           upscale_factor=2,
                           kernel_size=5):
        conv = conv_layer(in_channels,
                          out_channels * (upscale_factor ** 2),
                          kernel_size)
        pixel_shuffle = nn.PixelShuffle(upscale_factor)
        return sequential(conv, pixel_shuffle)

    def forward(self, x):
        conv1 = self.first_layer(x)
        conv2 = self.pixel_upsample(conv1)
        conv3 = self.pixel_upsample_2(conv2)
        return [conv1, conv2, conv3]


class HSIEncoder6(nn.Module):
    def __init__(self, in_ch=31, out_ch=64):
        super(HSIEncoder6, self).__init__()
        self.first_layer = Conv3XC(in_ch, out_ch, gain1=1, s=1, relu=True)
        self.pixel_upsample = self.pixelshuffle_block(out_ch, out_ch)
        self.pixel_upsample_2 = self.pixelshuffle_block(out_ch, out_ch, upscale_factor=3)

    def pixelshuffle_block(self,
                           in_channels,
                           out_channels,
                           upscale_factor=2,
                           kernel_size=5):
        conv = conv_layer(in_channels,
                          out_channels * (upscale_factor ** 2),
                          kernel_size)
        pixel_shuffle = nn.PixelShuffle(upscale_factor)
        return sequential(conv, pixel_shuffle)

    def forward(self, x):
        conv1 = self.first_layer(x)
        conv2 = self.pixel_upsample(conv1)
        conv3 = self.pixel_upsample_2(conv2)
        return [conv1, conv2, conv3]