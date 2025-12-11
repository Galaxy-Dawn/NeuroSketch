import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter


def gem(x, p=3, eps=1e-6, module=None):
    if module == 1:
        return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1),)).pow(1.0 / p)
    else:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False, module=2):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.module = module
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps, module = self.module)
        return ret

    def __repr__(self):
        return (self.__class__.__name__  + f"(p={self.p.data.tolist()[0]:.4f},eps={self.eps})")


class DepthwiseConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs):
        return self.conv(inputs)


class PointwiseConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )
    def forward(self, inputs):
        return self.conv(inputs)


class DepthwiseConv1dShared(nn.Module):
    def __init__(
            self,
            in_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = True,
    ) -> None:
        super(DepthwiseConv1dShared, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.shared_weight = nn.Parameter(torch.randn(1, 1, kernel_size))
        if bias:
            self.shared_bias = nn.Parameter(torch.zeros(in_channels))
        else:
            self.register_parameter('shared_bias', None)

    def forward(self, x):
        weight = self.shared_weight.repeat(self.in_channels, 1, 1)  # 形状: (in_channels, 1, kernel_size)
        if self.bias:
            bias = self.shared_bias
        else:
            bias = None
        return F.conv1d(x, weight, bias=bias, stride=self.stride, padding=self.padding, groups=self.in_channels)


