import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from timm.models.layers import trunc_normal_, DropPath
from models.submodels.DSCBlock import DSCBlks
from models.submodels.GaborBlock import GaborBlock
from models.submodels.ApproxGabor import SimpleGaborConv

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

class PalmMoE(nn.Module):
    """MoE实现 - 计算所有专家并加权求和"""

    def __init__(self, expert_blocks: List[nn.Module], input_dim: int,
                 temperature: float = 1.0, use_softmax: bool = True):
        """
        Args:
            expert_blocks: 专家模块列表
            input_dim: 输入特征维度
            temperature: softmax温度参数
            use_softmax: 是否使用softmax归一化权重
        """
        super().__init__()
        # self.common_ex1 = ConvNeXtBlock(input_dim, drop_path=0.)
        # self.common_ex2 = ConvNeXtBlock(input_dim, drop_path=0.1)
        # self.gabor_ex = GaborBlock(in_channels=input_dim,out_channels=input,num_blks=2,device = torch.device('cuda:0'))
        # self.snake_ex =  DSCBlks(
        #            snake_numbers=2,
        #            in_channels=input_dim,
        #            out_channels=input_dim,
        #            kernel_size=9,
        #            if_offset=True,
        #            extend_scope=1,
        #            device=torch.device("cuda:0"),
        #        )
        #
        self.experts = nn.ModuleList(expert_blocks)
        self.num_experts = len(expert_blocks)
        self.temperature = temperature
        self.use_softmax = use_softmax

        # Gating Network
        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, self.num_experts)
        )

        # 专家重要性权重（可选）
        self.expert_importance = nn.Parameter(torch.ones(self.num_experts))

        print(f"SimpleMoE初始化: {self.num_experts}个专家模块")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [batch_size, channels, height, width]

        Returns:
            output: 加权求和后的输出
        """
        batch_size = x.size(0)

        # 准备门控输入（全局平均池化）
        gate_input = F.adaptive_avg_pool2d(x, (1, 1)).view(batch_size, -1)

        # 计算原始门控权重
        raw_weights = self.gating_network(gate_input)  # [batch_size, num_experts]

        # 应用专家重要性权重
        weighted_logits = raw_weights * self.expert_importance.unsqueeze(0)

        # 可选：应用温度系数和softmax
        if self.use_softmax:
            gate_weights = F.softmax(weighted_logits / self.temperature, dim=-1)
        else:
            # 或者使用sigmoid + 归一化
            gate_weights = torch.sigmoid(weighted_logits)
            gate_weights = gate_weights / gate_weights.sum(dim=-1, keepdim=True)

        # 计算所有专家的输出
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)  # [batch_size, channels, height, width]
            expert_outputs.append(expert_out.unsqueeze(1))  # [batch_size, 1, channels, height, width]

        # 合并专家输出
        all_expert_outputs = torch.cat(expert_outputs, dim=1)  # [batch_size, num_experts, channels, height, width]

        # 应用门控权重
        gate_weights_expanded = gate_weights.view(batch_size, self.num_experts, 1, 1, 1)
        output = (all_expert_outputs * gate_weights_expanded).sum(dim=1)

        # 保存门控权重用于监控
        self.last_gate_weights = gate_weights.detach()

        return output

    def get_gate_statistics(self):
        """获取门控统计信息"""
        if hasattr(self, 'last_gate_weights'):
            gate_weights = self.last_gate_weights
            with torch.no_grad():
                # 平均专家使用概率
                expert_probs = gate_weights.mean(dim=0)
                # 门控权重熵（负载均衡指标）
                entropy = -torch.sum(gate_weights * torch.log(gate_weights + 1e-8), dim=-1)
                avg_entropy = entropy.mean()

            return {
                'expert_probabilities': expert_probs.cpu().numpy(),
                'average_entropy': avg_entropy.item(),
                'max_expert_usage': expert_probs.max().item(),
                'min_expert_usage': expert_probs.min().item()
            }
        return None


def PalmMoe_Constructor_base(input_dim, expert_nums=[1,1,1,1], device='cuda:1'):
    experts = []
    for i in range(expert_nums[0]):
        experts.append(
            ConvNeXtBlock(input_dim, drop_path=0.)
        )
    for i in range(expert_nums[2]):
        experts.append(
            SimpleGaborConv(in_channels=input_dim,out_channels=input_dim,kernel_size=9, orientations=4,device=device)
        )
    for i in range(expert_nums[3]):
        experts.append(
            DSCBlks(
                snake_numbers=1,
                in_channels=input_dim,
                out_channels=input_dim,
                kernel_size=9,
                if_offset=True,
                extend_scope=1,
                device=device,
            )
        )
    for i in range(expert_nums[1]):
        experts.append(
            TransformerBlock(channels=input_dim, expansion_factor=1)
        )
    return PalmMoE(experts, input_dim=input_dim)

def PalmMoe_Constructor(input_dim, expert_nums=[1,1,1,1],device='cuda:1'):
    experts = []
    for i in range(expert_nums[0]):
        experts.append(
            ConvNeXtBlock(input_dim, drop_path=0.)
        )
    for i in range(expert_nums[1]):
        experts.append(
            ConvNeXtBlock(input_dim, drop_path=0.1)
        )
    for i in range(expert_nums[2]):
        experts.append(
            SimpleGaborConv(in_channels=input_dim,out_channels=input_dim,kernel_size=9, orientations=4,device=device)
        )
    for i in range(expert_nums[3]):
        experts.append(
            DSCBlks(
                snake_numbers=1,
                in_channels=input_dim,
                out_channels=input_dim,
                kernel_size=9,
                if_offset=True,
                extend_scope=1,
                device=device,
            )
        )
    return PalmMoE(experts, input_dim=input_dim)


# 使用示例
if __name__ == "__main__":

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    # 创建简单MoE
    moe = PalmMoe_Constructor(input_dim=64,expert_nums=[1,1,1,1],device=device).to(device)

    # 测试
    x = torch.randn(1, 64, 32, 32).to(device)
    output = moe(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    # 查看门控统计
    stats = moe.get_gate_statistics()
    if stats:
        print(f"专家使用概率: {stats['expert_probabilities']}")
        print(f"平均熵: {stats['average_entropy']:.4f}")

    # 梯度测试
    target = torch.randn_like(output)
    loss = F.mse_loss(output, target)
    loss.backward()

    print("梯度测试通过!")