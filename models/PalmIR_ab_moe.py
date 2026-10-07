import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
import numpy.random as random
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PalmMoE import PalmMoe_Constructor, PalmMoe_Constructor_base

class DownSample(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(in_channels,out_channels,3,2,1),
                                  nn.LeakyReLU(0.2),)

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(nn.ConvTranspose2d(in_channels,out_channels,4,2,1),
                                  nn.LeakyReLU(0.2),)
    def forward(self, x):
        return self.body(x)

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNeXtBlock(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class MDTA(nn.Module):
    def __init__(self, channels, num_heads=4):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads=4, expansion_factor=2.66):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w))
        return x

class PalmIR_Ablation(nn.Module):
    def __init__(self, channels = [64,128,256,512]):
        super(PalmIR_Ablation, self).__init__()
        # first down
        self.stage_dn0 = nn.Conv2d(3,channels[0],3,2,1)
        # shallow feature
        self.d_stage_1 = ConvNeXtBlock(channels[0],0.1)
        # output stage 1 feature
        self.d_stage_ds1 = DownSample(channels[0],channels[1])
        #
        #
        self.d_stage_2 = TransformerBlock(channels=channels[1],expansion_factor=2)
        # output stage 2 feature
        self.d_stage_ds2 = DownSample(channels[1],channels[2])
        #
        #
        self.d_stage_3 = TransformerBlock(channels=channels[2], expansion_factor=2)
        # output stage 3 feature
        self.d_stage_ds3 = DownSample(channels[2],channels[3])
        #
        #
        self.bottleneck_trans = TransformerBlock(channels=channels[3], expansion_factor=1)
        #
        #
        # cat stage 3 feature ->1*1 conv
        self.u_stage_us3 = UpSample(channels[3],channels[2])
        self.stage3_conv1 = nn.Conv2d(channels[2]*2,channels[2],1,1,0)
        self.u_stage_3 = TransformerBlock(channels=channels[2], expansion_factor=2)
        #
        #
        self.u_stage_us2 = UpSample(channels[2],channels[1])
        # cat stage 2 feature
        self.stage2_conv1 = nn.Conv2d(channels[1]*2,channels[1],1,1,0)
        self.u_stage_2 = TransformerBlock(channels=channels[1], expansion_factor=2)
        #
        #
        self.u_stage_us1 = UpSample(channels[1],channels[0])
        # cat stage 1 feature
        self.stage1_conv1 = nn.Conv2d(channels[0]*2,channels[0],1,1,0)
        self.u_stage_1 = TransformerBlock(channels=channels[0], expansion_factor=2)
        #
        # last up
        self.u_stage_us0 = UpSample(channels[0], channels[0])
        #
        self.final = nn.Sequential(
            nn.Conv2d(channels[0], 3, 1, 1, 0),
            nn.Sigmoid(),
        )
    def forward(self, x): # (1,3,128,128)
        x = self.stage_dn0(x) # (1,64,64,64)

        x = self.d_stage_1(x) # (1,64,64,64)
        x_stage1 = x # (1,64,64,64)
        x = self.d_stage_ds1(x) # (1,128,32,32)

        x = self.d_stage_2(x) # (1,128,32,32)
        x_stage2 = x  # (1,128,32,32)
        x = self.d_stage_ds2(x) # (1,256,16,16)

        x = self.d_stage_3(x) # (1,256,16,16)
        x_stage3 = x
        x = self.d_stage_ds3(x) # (1,512,8,8)

        x = self.bottleneck_trans(x)

        x = self.u_stage_us3(x) # (1,256,16,16)
        cat3 = self.stage3_conv1(torch.cat((x, x_stage3), dim=1))
        x = self.u_stage_3(cat3) # (1,256,16,16)

        x = self.u_stage_us2(x)
        cat2 = self.stage2_conv1(torch.cat((x, x_stage2), dim=1))
        x = self.u_stage_2(cat2)

        x = self.u_stage_us1(x)
        cat1 = self.stage1_conv1(torch.cat((x, x_stage1), dim=1))
        x = self.u_stage_1(cat1)

        # last
        x = self.u_stage_us0(x)
        x = self.final(x)

        return x


class PalmIR_ab_moe(nn.Module):
    def __init__(self, channels = [64,128,256,512], device=torch.device('cpu')):
        super(PalmIR_ab_moe, self).__init__()
        # first down
        self.stage_dn0 = nn.Conv2d(3,channels[0],3,2,1)
        # shallow feature
        # self.d_stage_1 = ConvNeXtBlock(channels[0],0.1)
        # stage_1_moe
        self.d_stage_1 = PalmMoe_Constructor(channels[0],[2,2,0,0], device=device)
        # output stage 1 feature ->
        self.d_stage_ds1 = DownSample(channels[0],channels[1])
        #
        #
        self.d_stage_2 = TransformerBlock(channels=channels[1],expansion_factor=2.66)
        # stage 2 moe
        self.moe_2 = PalmMoe_Constructor(channels[1], [1, 1, 0, 0], device=device)
        # output stage 2 feature ->
        self.d_stage_ds2 = DownSample(channels[1],channels[2])
        #
        #
        self.d_stage_3 = TransformerBlock(channels=channels[2], expansion_factor=2.66)
        # output stage 3 feature
        #
        self.moe_3 = PalmMoe_Constructor(channels[2], [1, 1, 0, 0], device=device)
        #
        self.d_stage_ds3 = DownSample(channels[2],channels[3])
        #
        #
        self.bottleneck_trans = TransformerBlock(channels=channels[3], expansion_factor=2)
        #
        #
        # cat stage 3 feature ->1*1 conv
        self.u_stage_us3 = UpSample(channels[3],channels[2])
        self.stage3_conv1 = nn.Conv2d(channels[2]*2,channels[2],1,1,0)
        self.u_stage_3 = TransformerBlock(channels=channels[2], expansion_factor=2)
        #
        #
        self.u_stage_us2 = UpSample(channels[2],channels[1])
        # cat stage 2 feature
        self.stage2_conv1 = nn.Conv2d(channels[1]*2,channels[1],1,1,0)
        self.u_stage_2 = TransformerBlock(channels=channels[1], expansion_factor=2)
        #
        #
        self.u_stage_us1 = UpSample(channels[1],channels[0])
        # cat stage 1 feature
        self.stage1_conv1 = nn.Conv2d(channels[0]*2,channels[0],1,1,0)
        self.u_stage_1 = TransformerBlock(channels=channels[0], expansion_factor=2)
        #
        # last up
        self.u_stage_us0 = UpSample(channels[0], channels[0])
        #
        self.final = nn.Sequential(
            nn.Conv2d(channels[0], 3, 1, 1, 0),
            nn.Sigmoid(),
        )
    def forward(self, x): # (1,3,128,128)
        x = self.stage_dn0(x) # (1,64,64,64)

        x = self.d_stage_1(x) # (1,64,64,64)
        # x = self.moe_1(x)
        x_stage1 = x # (1,64,64,64)
        x = self.d_stage_ds1(x) # (1,128,32,32)

        x = self.d_stage_2(x) # (1,128,32,32)
        x = self.moe_2(x)
        x_stage2 = x  # (1,128,32,32)
        x = self.d_stage_ds2(x) # (1,256,16,16)

        x = self.d_stage_3(x) # (1,256,16,16)
        x = self.moe_3(x)
        x_stage3 = x
        x = self.d_stage_ds3(x) # (1,512,8,8)

        x = self.bottleneck_trans(x)

        x = self.u_stage_us3(x) # (1,256,16,16)
        cat3 = self.stage3_conv1(torch.cat((x, x_stage3), dim=1))
        x = self.u_stage_3(cat3) # (1,256,16,16)

        x = self.u_stage_us2(x)
        cat2 = self.stage2_conv1(torch.cat((x, x_stage2), dim=1))
        x = self.u_stage_2(cat2)

        x = self.u_stage_us1(x)
        cat1 = self.stage1_conv1(torch.cat((x, x_stage1), dim=1))
        x = self.u_stage_1(cat1)

        # last
        x = self.u_stage_us0(x)
        x = self.final(x)

        return x

class PalmIR_large(nn.Module):
    def __init__(self, channels = [64,128,256,512], device=torch.device('cpu')):
        super(PalmIR_large, self).__init__()
        # first down
        self.stage_dn0 = nn.Conv2d(3,channels[0],3,2,1)
        # shallow feature
        # self.d_stage_1 = ConvNeXtBlock(channels[0],0.1)
        # stage_1_moe
        self.d_stage_1 = PalmMoe_Constructor_base(channels[0],[1,1,1,0], device=device)
        # output stage 1 feature ->
        self.d_stage_ds1 = DownSample(channels[0],channels[1])
        #
        #
        self.d_stage_2_before = TransformerBlock(channels=channels[1],expansion_factor=2.66)
        # stage 2 moe
        self.moe_2 = PalmMoe_Constructor(channels[1], [1, 1, 1, 1], device=device)
        # output stage 2 feature ->
        self.d_stage_2_after = TransformerBlock(channels=channels[1], expansion_factor=2.66)
        self.d_stage_ds2 = DownSample(channels[1],channels[2])
        #
        #
        self.d_stage_3 = TransformerBlock(channels=channels[2], expansion_factor=2.66)
        # output stage 3 feature
        #
        self.moe_3 = PalmMoe_Constructor(channels[2], [1, 1, 1, 1], device=device)
        #
        self.d_stage_ds3 = DownSample(channels[2],channels[3])
        #
        #
        self.bottleneck_trans = nn.Sequential(
            TransformerBlock(channels=channels[3], expansion_factor=2),
        )
        #
        #
        # cat stage 3 feature ->1*1 conv
        self.u_stage_us3 = UpSample(channels[3],channels[2])
        self.stage3_conv1 = nn.Conv2d(channels[2]*2,channels[2],1,1,0)
        self.u_stage_3 = TransformerBlock(channels=channels[2], expansion_factor=2)
        #
        #
        self.u_stage_us2 = UpSample(channels[2],channels[1])
        # cat stage 2 feature
        self.stage2_conv1 = nn.Conv2d(channels[1]*2,channels[1],1,1,0)
        self.u_stage_2 = TransformerBlock(channels=channels[1], expansion_factor=2)
        #
        #
        self.u_stage_us1 = UpSample(channels[1],channels[0])
        # cat stage 1 feature
        self.stage1_conv1 = nn.Conv2d(channels[0]*2,channels[0],1,1,0)
        self.u_stage_1 = TransformerBlock(channels=channels[0], expansion_factor=2)
        #
        # last up
        self.u_stage_us0 = UpSample(channels[0], channels[0])
        #
        self.final = nn.Sequential(
            nn.Conv2d(channels[0], 3, 1, 1, 0),
        )
    def forward(self, x): # (1,3,128,128)
        x = self.stage_dn0(x) # (1,64,64,64)

        x = self.d_stage_1(x) # (1,64,64,64)
        # x = self.moe_1(x)
        x_stage1 = x # (1,64,64,64)
        x = self.d_stage_ds1(x) # (1,128,32,32)

        x = self.d_stage_2_before(x) # (1,128,32,32)
        x = self.moe_2(x)
        x = self.d_stage_2_after(x)
        x_stage2 = x  # (1,128,32,32)
        x = self.d_stage_ds2(x) # (1,256,16,16)

        x = self.d_stage_3(x) # (1,256,16,16)
        x = self.moe_3(x)
        x_stage3 = x
        x = self.d_stage_ds3(x) # (1,512,8,8)

        x = self.bottleneck_trans(x)

        x = self.u_stage_us3(x) # (1,256,16,16)
        cat3 = self.stage3_conv1(torch.cat((x, x_stage3), dim=1))
        x = self.u_stage_3(cat3) # (1,256,16,16)

        x = self.u_stage_us2(x)
        cat2 = self.stage2_conv1(torch.cat((x, x_stage2), dim=1))
        x = self.u_stage_2(cat2)

        x = self.u_stage_us1(x)
        cat1 = self.stage1_conv1(torch.cat((x, x_stage1), dim=1))
        x = self.u_stage_1(cat1)

        # last
        x = self.u_stage_us0(x)
        x = self.final(x)

        return x


if __name__ == '__main__':
    import time
    from thop import profile
    device = torch.device('cuda:1')
    x = torch.randn(8, 3, 128, 128).to(device)
    model = PalmIR_large(channels=[64,128,256,512],device=device).to(device)
    flops, params = profile(model, inputs=(x,))
    print(f"原始结果: {flops:,} FLOPs, {params:,} 参数")
    start_time = time.time()
    y = model(x)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"总推理时间: {total_time:.4f}秒")
    print(x.shape)
