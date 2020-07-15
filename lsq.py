"""
    LSQ: Learned Step Size Quantization
    ICLR2020 open review
    https://quanoview.readthedocs.io/en/latest/_raw/LSQ.html
"""
import torch
import torch.nn.functional as F
import math
from models.modules import _Conv2dQ, Qmodes, _LinearQ, _ActQ

import ipdb

__all__ = ['Conv2dLSQ', 'LinearLSQ', 'ActLSQ']


class FunLSQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp):
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp
        q_w = (weight / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        q_w = weight / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = torch.ones(indicate_small.shape).to(indicate_small.device) - indicate_small - indicate_big
        grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                -q_w + q_w.round())) * grad_weight / g).sum().unsqueeze(dim=0)
        grad_weight = indicate_middle * grad_weight
        # The following operation can make sure that alpha is always greater than zero in any case and can also
        # suppress the update speed of alpha. (Personal understanding)
        grad_alpha.clamp_(-alpha.item(), alpha.item())  # FYI
        return grad_weight, grad_alpha, None, None, None


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad


class Conv2dLSQ(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits=4,
                 mode=Qmodes.layer_wise):
        super(Conv2dLSQ, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits, mode=mode)

    def forward(self, x):
        if self.nbits < 0:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        # w_reshape = self.weight.reshape([self.weight.shape[0], -1]).transpose(0, 1)
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(self.weight.abs().max() / 2 ** (self.nbits - 1))

            # self.alpha.data.copy_(self.weight.abs().max() * 2)
            self.init_state.fill_(1)
        

        ## Pslease see STE_LSQ.ipynb for detailed comparison.
        """ Method1: 
        alpha = grad_scale(self.alpha, self.alpha_scale)
        w = self.weight / alpha
        w = w.clamp(-(2 ** (self.nbits - 1)), (2 ** (self.nbits - 1) - 1))
        q_w = round_pass(w)
        w_q = q_w * alpha
        """

        # Method2:
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        g = math.sqrt(self.weight.numel() * Qp)
        w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        # wq = y.transpose(0, 1).reshape(self.weight.shape).detach() + self.weight - self.weight.detach()
        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class LinearLSQ(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits=4):
        super(_LinearQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias, nbits=nbits)

    def forward(self, x):
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(self.weight.abs().max() / 2 ** (self.nbits - 1))
            self.init_state.fill_(1)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        g = math.sqrt(self.weight.numel() * Qp)
        w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        return F.conv2d(input, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ActLSQ(_ActQ):
    def __init(self, nbits=4, sign=False):
        super(ActLSQ, self).__init(nbits=nbits, sign=sign)

    def forward(self, x):
        if self.alpha is None:
            return x
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(x.max() / 2 ** (self.nbits - 1) * 0.25)
            self.init_state.fill_(1)
        if self.sign:
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.nbits - 1
        g = math.sqrt(x.numel() * Qp)
        x_q = FunLSQ.apply(x, self.alpha, g, Qn, Qp)
        return x_q
