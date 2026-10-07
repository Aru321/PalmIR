import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from models.submodels.suboprs.GaborConv import GaborConv2d

class GaborBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blks=2, device = torch.device('cuda:0')):
        super(GaborBlock, self).__init__()
        self.gabor_group = nn.ModuleList()
        for i in range(num_blks):
            self.gabor_group.append(GaborConv2d(in_channels, in_channels, kernel_size=(3, 3), padding=1, device=device))
        self.gabor_group.append(GaborConv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1, device=device))
        self.relu = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        for i,layer in enumerate(self.gabor_group):
            x = layer(x)
        x = self.relu(x)
        return x

if __name__ == "__main__":

    x = torch.randn(1, 32, 32, 32).cuda()
    model = GaborBlock(32, 64, num_blks=2, device = torch.device('cuda:0')).cuda()
    y = model(x)
    print(y.shape)