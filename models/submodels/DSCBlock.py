# -*- coding: utf-8 -*-
import torch
from torch import nn, cat
from torch.nn.functional import dropout
from models.submodels.suboprs.SnakeConv import DSConv_pro

class EncoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x

class DSCBlk(nn.Module):
    def __init__(
        self,
        n_channels,
        out_channels,
        kernel_size,
        extend_scope,
        if_offset,
        device,
    ):
        """
        Our DSCNet
        :param n_channels: input channel
        :param n_classes: output channel
        :param kernel_size: the size of kernel
        :param extend_scope: the range to expand (default 1 for this method)
        :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
        :param device: set on gpu
        :param number: basic layer numbers
        :param dim:
        """
        super().__init__()
        device = device
        # 初始conv
        self.conv0 = EncoderConv(n_channels,n_channels)
        #
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.if_offset = if_offset
        self.relu = nn.ReLU(inplace=True)


        self.conv0x = DSConv_pro(
            n_channels,
            n_channels,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )

        self.conv0y = DSConv_pro(
            n_channels,
            n_channels,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )

        self.conv1 = nn.Conv2d(in_channels=n_channels * 3, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                               bias=False)

    def forward(self, x):
        # block0
        x_00_0 = self.conv0(x)
        x_0x_0 = self.conv0x(x)
        x_0y_0 = self.conv0y(x)
        out = self.conv1(torch.cat([x_00_0, x_0x_0, x_0y_0], dim=1))

        return out

class DSCBlks(nn.Module):
    def __init__(
            self,
            snake_numbers,
            in_channels,
            out_channels,
            kernel_size,
            extend_scope=1,
            if_offset=True,
            device="cuda:0"
    ):
        super().__init__()

        # 为了保持尺寸，padding = 2 * (3-1)//2 = 2
        self.dilated_conv = nn.Conv2d(in_channels,in_channels, kernel_size=3, padding=2, dilation=2)
        #
        self.Snakes = nn.ModuleList()
        for i in range(snake_numbers):
            self.Snakes.append(DSCBlk(
                in_channels,
                in_channels,
                kernel_size,
                extend_scope,
                if_offset,
                device,
            ))
        self.Snakes.append(DSCBlk(
            in_channels,
            out_channels,
            kernel_size,
            extend_scope,
            if_offset,
            device,
        ))
    def forward(self, x):
        x = self.dilated_conv(x)
        for i,layer in enumerate(self.Snakes):
            x = layer(x)
        return x

if __name__ == "__main__":
    x = torch.randn(1, 32, 64, 64).to('cuda:1')
    blk = DSCBlks(
        snake_numbers=2,
        in_channels=32,
        out_channels=64,
        kernel_size=9,
        if_offset=True,
        extend_scope=1,
        device='cuda:1',
    ).to('cuda:1')

    res = blk(x)
    print(res.shape)