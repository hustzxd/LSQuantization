"""
    Quantized modules: the base class
"""
import torch
import torch.nn as nn
from enum import Enum
from torch.nn.parameter import Parameter

__all__ = ['Qmodes', '_Conv2dQ', '_LinearQ', '_ActQ']


class Qmodes(Enum):
    layer_wise = 1
    kernel_wise = 2


class _Conv2dQ(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits=4,
                 mode=Qmodes.kernel_wise):
        super(_Conv2dQ, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                       padding=padding, dilation=dilation, groups=groups, bias=bias)
        if nbits < 0:
            self.register_parameter('alpha', None)
            return
        self.nbits = nbits
        self.q_mode = mode
        if mode == Qmodes.kernel_wise:
            self.alpha = Parameter(torch.Tensor(out_channels))
            self.is_layer_wise = False
        else:  # layer-wise quantization
            self.alpha = Parameter(torch.Tensor(1))
            self.is_layer_wise = True
        self.register_buffer('init_state', torch.zeros(1))

    def extra_repr(self):
        s_prefix = super(_Conv2dQ, self).extra_repr()
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)
        return '{}, nbits={}, qmode={}'.format(s_prefix, self.nbits, self.q_mode, )


class _LinearQ(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, nbits=4):
        super(_LinearQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        if nbits < 0:
            self.register_parameter('alpha', None)
            return
        self.nbits = nbits
        self.alpha = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))

    def extra_repr(self):
        s_prefix = super(_LinearQ, self).extra_repr()
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)
        return '{}, nbits={}'.format(s_prefix, self.nbits)


class _ActQ(nn.Module):
    def __init__(self, nbits=4, sign=False):
        super(_ActQ, self).__init__()
        if nbits < 0:
            self.register_parameter('alpha', None)
            return
        self.nbits = nbits
        self.sign = sign
        self.alpha = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))

    def extra_repr(self):
        s_prefix = super(_ActQ, self).extra_repr()
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)
        return '{}, nbits={}, sign={}'.format(s_prefix, self.nbits, self.sign)
