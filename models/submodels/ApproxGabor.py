import torch
from torch import nn
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class SimpleGaborConv(nn.Module):
    """更简洁的Gabor卷积实现"""

    def __init__(self, in_channels, out_channels, kernel_size=15, orientations=4, device='cpu'):
        super().__init__()
        self.device = device
        # 预计算Gabor核
        self.kernels = self._create_gabor_kernels(kernel_size, orientations)
        self.padding = kernel_size // 2

        # 后续卷积
        self.conv = nn.Conv2d(in_channels * orientations, out_channels, 3, padding=1)

    def _create_gabor_kernels(self, kernel_size, orientations):
        kernels = []

        for i in range(orientations):
            theta = i * math.pi / orientations

            # 创建坐标张量
            x = torch.linspace(-1, 1, kernel_size)
            y = torch.linspace(-1, 1, kernel_size)
            X, Y = torch.meshgrid(x, y, indexing='ij')

            rotX = X * torch.cos(torch.tensor(theta)) + Y * torch.sin(torch.tensor(theta))
            rotY = -X * torch.sin(torch.tensor(theta)) + Y * torch.cos(torch.tensor(theta))

            sigma = 0.5
            gabor = torch.exp(-0.5 * (rotX ** 2 + rotY ** 2) / sigma ** 2)
            gabor = gabor * torch.cos(2 * math.pi * rotX / 0.3)  # 现在rotX是张量

            # 归一化
            gabor = (gabor - gabor.mean()) / (gabor.std() + 1e-8)
            kernels.append(gabor.unsqueeze(0))

        return torch.stack(kernels).to(self.device)  # [orientations, 1, H, W]

    def forward(self, x):
        batch_size, channels, h, w = x.shape
        outputs = []

        for c in range(channels):
            channel_feats = []
            for kernel in self.kernels:
                filtered = F.conv2d(x[:, c:c + 1], kernel.unsqueeze(0),
                                    padding=self.padding)
                channel_feats.append(filtered)
            outputs.append(torch.cat(channel_feats, dim=1))

        # 合并所有通道的Gabor特征
        gabor_features = torch.cat(outputs, dim=1)

        # 应用后续卷积
        return torch.relu(self.conv(gabor_features))

if __name__ == '__main__':
    device = 'cuda:1'
    tensor = torch.randn(1, 64, 64, 64).to(device)
    model = SimpleGaborConv(64, 64, kernel_size=9, orientations=4, device=device).to(device)
    print(model(tensor).shape)