from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch

def sequential(*args):
    """
    Modules will be added to the a Sequential Container in the order they
    are passed.

    Parameters
    ----------
    args: Definition of Modules in order.
    -------
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                'sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value


def conv_layer(in_channels,
               out_channels,
               kernel_size,
               bias=True):
    """
    Re-write convolution layer for adaptive `padding`.
    """
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2),
               int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias)


class Conv3XC(nn.Module):
    def __init__(self, c_in, c_out, gain1=1, s=1, padding=0, bias=True, relu=False):
        super(Conv3XC, self).__init__()
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.stride = s
        self.has_relu = relu
        gain = gain1
        self.padding = padding

        self.sk = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=1, padding=0, stride=s, bias=bias)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_in * gain, kernel_size=1, padding=0, bias=bias),
            nn.Conv2d(in_channels=c_in * gain, out_channels=c_out * gain, kernel_size=3, stride=s, padding=0, bias=bias),
            nn.Conv2d(in_channels=c_out * gain, out_channels=c_out, kernel_size=1, padding=0, bias=bias),
        )

        self.eval_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, stride=s, bias=bias)
        self.eval_conv.weight.requires_grad = False
        self.eval_conv.bias.requires_grad = False
        self.update_params()

    def update_params(self):
        w1 = self.conv[0].weight.data.clone().detach()
        b1 = self.conv[0].bias.data.clone().detach()
        w2 = self.conv[1].weight.data.clone().detach()
        b2 = self.conv[1].bias.data.clone().detach()
        w3 = self.conv[2].weight.data.clone().detach()
        b3 = self.conv[2].bias.data.clone().detach()

        w = F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        self.weight_concat = F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        sk_w = self.sk.weight.data.clone().detach()
        sk_b = self.sk.bias.data.clone().detach()
        target_kernel_size = 3

        H_pixels_to_pad = (target_kernel_size - 1) // 2
        W_pixels_to_pad = (target_kernel_size - 1) // 2
        sk_w = F.pad(sk_w, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])

        self.weight_concat = self.weight_concat + sk_w
        self.bias_concat = self.bias_concat + sk_b

        self.eval_conv.weight.data = self.weight_concat
        self.eval_conv.bias.data = self.bias_concat

    def forward(self, x):
        if self.training:
            pad = 1
            x_pad = F.pad(x, (pad, pad, pad, pad), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            self.update_params()
            out = self.eval_conv(x)

        if self.has_relu:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out


BatchNorm3d = nn.BatchNorm3d


class BNReLUConv3d(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=False):
        super(BNReLUConv3d, self).__init__()
        self.add_module('bn', BatchNorm3d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))
        self.add_module('conv', nn.Conv3d(in_channels, channels, k, s, p, bias=False))


class BNReLUDeConv3d(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=False):
        super(BNReLUDeConv3d, self).__init__()
        self.add_module('bn', BatchNorm3d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))
        self.add_module('deconv', nn.ConvTranspose3d(in_channels, channels, k, s, p, bias=False))


class BNReLUUpsampleConv3d(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, upsample=(1, 2, 2), inplace=False):
        super(BNReLUUpsampleConv3d, self).__init__()
        self.add_module('bn', BatchNorm3d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))
        self.add_module('upsample_conv', UpsampleConv3d(in_channels, channels, k, s, p, bias=False, upsample=upsample))


class UpsampleConv3d(nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, upsample=None):
        super(UpsampleConv3d, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=upsample, mode='trilinear', align_corners=True)

        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.conv3d(x_in)
        return out


class BasicConv3d(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, bias=False, bn=True):
        super(BasicConv3d, self).__init__()
        if bn:
            self.add_module('bn', BatchNorm3d(in_channels))
        self.add_module('conv', nn.Conv3d(in_channels, channels, k, s, p, bias=bias))


class BasicDeConv3d(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, bias=False, bn=True):
        super(BasicDeConv3d, self).__init__()
        if bn:
            self.add_module('bn', BatchNorm3d(in_channels))
        self.add_module('deconv', nn.ConvTranspose3d(in_channels, channels, k, s, p, bias=bias))


class BasicUpsampleConv3d(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, upsample=(1, 2, 2), bn=True):
        super(BasicUpsampleConv3d, self).__init__()
        if bn:
            self.add_module('bn', BatchNorm3d(in_channels))
        self.add_module('upsample_conv', UpsampleConv3d(in_channels, channels, k, s, p, bias=False, upsample=upsample))


class QRNN3DLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, conv_layer, act='tanh'):
        super(QRNN3DLayer, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        # quasi_conv_layer
        self.conv = conv_layer
        self.act = act

    def _conv_step(self, inputs):
        gates = self.conv(inputs)
        Z, F = gates.split(split_size=self.hidden_channels, dim=1)
        if self.act == 'tanh':
            return Z.tanh(), F.sigmoid()
        elif self.act == 'relu':
            return Z.relu(), F.sigmoid()
        elif self.act == 'none':
            return Z, F.sigmoid
        else:
            raise NotImplementedError

    def _rnn_step(self, z, f, h):
        # uses 'f pooling' at each time step
        h_ = (1 - f) * z if h is None else f * h + (1 - f) * z
        return h_

    def forward(self, inputs, reverse=False):
        h = None
        Z, F = self._conv_step(inputs)
        h_time = []

        if not reverse:
            for time, (z, f) in enumerate(zip(Z.split(1, 2), F.split(1, 2))):  # split along timestep
                h = self._rnn_step(z, f, h)
                h_time.append(h)
        else:
            for time, (z, f) in enumerate((zip(
                    reversed(Z.split(1, 2)), reversed(F.split(1, 2))
            ))):  # split along timestep
                h = self._rnn_step(z, f, h)
                h_time.insert(0, h)

        # return concatenated hidden states
        return torch.cat(h_time, dim=2)

    def extra_repr(self):
        return 'act={}'.format(self.act)


class BiQRNN3DLayer(QRNN3DLayer):
    def _conv_step(self, inputs):
        gates = self.conv(inputs)
        Z, F1, F2 = gates.split(split_size=self.hidden_channels, dim=1)
        if self.act == 'tanh':
            return Z.tanh(), F1.sigmoid(), F2.sigmoid()
        elif self.act == 'relu':
            return Z.relu(), F1.sigmoid(), F2.sigmoid()
        elif self.act == 'none':
            return Z, F1.sigmoid(), F2.sigmoid()
        else:
            raise NotImplementedError

    def forward(self, inputs, fname=None):
        h = None
        Z, F1, F2 = self._conv_step(inputs)
        hsl = [];
        hsr = []
        zs = Z.split(1, 2)

        for time, (z, f) in enumerate(zip(zs, F1.split(1, 2))):  # split along timestep
            h = self._rnn_step(z, f, h)
            hsl.append(h)

        h = None
        for time, (z, f) in enumerate((zip(
                reversed(zs), reversed(F2.split(1, 2))
        ))):  # split along timestep
            h = self._rnn_step(z, f, h)
            hsr.insert(0, h)

        # return concatenated hidden states
        hsl = torch.cat(hsl, dim=2)
        hsr = torch.cat(hsr, dim=2)

        if fname is not None:
            stats_dict = {'z': Z, 'fl': F1, 'fr': F2, 'hsl': hsl, 'hsr': hsr}
            torch.save(stats_dict, fname)
        return hsl + hsr


class BiQRNNConv3D(BiQRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bn=True, act='tanh'):
        super(BiQRNNConv3D, self).__init__(
            in_channels, hidden_channels, BasicConv3d(in_channels, hidden_channels * 3, k, s, p, bn=bn), act=act)


class BiQRNNDeConv3D(BiQRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bias=False, bn=True, act='tanh'):
        super(BiQRNNDeConv3D, self).__init__(
            in_channels, hidden_channels, BasicDeConv3d(in_channels, hidden_channels * 3, k, s, p, bias=bias, bn=bn),
            act=act)


class QRNNConv3D(QRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bn=True, act='tanh'):
        super(QRNNConv3D, self).__init__(
            in_channels, hidden_channels, BasicConv3d(in_channels, hidden_channels * 2, k, s, p, bn=bn), act=act)


class QRNNDeConv3D(QRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bn=True, act='tanh'):
        super(QRNNDeConv3D, self).__init__(
            in_channels, hidden_channels, BasicDeConv3d(in_channels, hidden_channels * 2, k, s, p, bn=bn), act=act)


class QRNNUpsampleConv3d(QRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, upsample=(1, 2, 2), bn=True, act='tanh'):
        super(QRNNUpsampleConv3d, self).__init__(
            in_channels, hidden_channels,
            BasicUpsampleConv3d(in_channels, hidden_channels * 2, k, s, p, upsample, bn=bn), act=act)
