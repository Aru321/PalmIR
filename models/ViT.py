import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: 输入张量 (B, N, C)，其中N=H*W
            mask: 注意力掩码 (可选)
        Returns:
            输出张量 (B, N, C)
        """
        B, N, C = x.shape

        # 生成Q, K, V
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(B, N, C)

        # 输出投影
        out = self.output_proj(out)
        return out


class PositionalEncoding2D(nn.Module):
    """2D位置编码"""

    def __init__(self, embed_dim, max_h=256, max_w=256):
        super().__init__()
        self.embed_dim = embed_dim

        # 创建位置编码
        pos_encoding_h = self._get_1d_position_encoding(max_h, embed_dim // 2)
        pos_encoding_w = self._get_1d_position_encoding(max_w, embed_dim // 2)

        self.register_buffer('pos_encoding_h', pos_encoding_h)
        self.register_buffer('pos_encoding_w', pos_encoding_w)

    def _get_1d_position_encoding(self, length, dim):
        """生成1D位置编码"""
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))

        pos_encoding = torch.zeros(length, dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding

    def forward(self, x):
        """
        Args:
            x: 输入张量 (B, C, H, W)
        Returns:
            添加位置编码后的张量 (B, C, H, W)
        """
        B, C, H, W = x.shape

        # 获取对应尺寸的位置编码
        pos_h = self.pos_encoding_h[:H].unsqueeze(1).unsqueeze(0).unsqueeze(0)  # (1, 1, H, 1, C//2)
        pos_w = self.pos_encoding_w[:W].unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, W, C//2)

        # 合并H和W的位置编码
        pos_encoding = torch.cat([
            pos_h.repeat(1, 1, 1, W, 1),
            pos_w.repeat(1, 1, H, 1, 1)
        ], dim=-1)  # (1, 1, H, W, C)

        # 调整维度顺序并添加到输入
        pos_encoding = pos_encoding.permute(0, 1, 4, 2, 3).squeeze(0)  # (1, C, H, W)
        return x + pos_encoding


class FeedForward(nn.Module):
    """前馈网络"""

    def __init__(self, embed_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer Block for Image Feature Extraction"""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0,
                 layer_norm_eps=1e-5, with_pos_encoding=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.with_pos_encoding = with_pos_encoding

        # 位置编码
        if with_pos_encoding:
            self.pos_encoding = PositionalEncoding2D(embed_dim)

        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

        # 注意力机制
        self.attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # 前馈网络
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.ffn = FeedForward(
            embed_dim=embed_dim,
            hidden_dim=mlp_hidden_dim,
            dropout=dropout
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: 输入张量 (B, C, H, W)
        Returns:
            输出张量 (B, C, H, W)
        """
        B, C, H, W = x.shape

        # 添加位置编码
        if self.with_pos_encoding:
            x = self.pos_encoding(x)

        # 重塑为序列格式: (B, C, H, W) -> (B, H*W, C)
        x_reshaped = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # 自注意力 + 残差连接
        residual = x_reshaped
        x_reshaped = self.norm1(x_reshaped)
        attn_out = self.attention(x_reshaped)
        x_reshaped = residual + self.dropout(attn_out)

        # 前馈网络 + 残差连接
        residual = x_reshaped
        x_reshaped = self.norm2(x_reshaped)
        ffn_out = self.ffn(x_reshaped)
        x_reshaped = residual + self.dropout(ffn_out)

        # 重塑回图像格式: (B, H*W, C) -> (B, C, H, W)
        out = x_reshaped.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return out


class TransformerEncoder(nn.Module):
    """Transformer编码器，包含多个Transformer Block"""

    def __init__(self, embed_dim, num_heads, num_layers, mlp_ratio=4.0,
                 dropout=0.0, with_pos_encoding=True):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                with_pos_encoding=with_pos_encoding if i == 0 else False
            ) for i in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# 使用示例
if __name__ == "__main__":
    # 创建Transformer Block
    transformer = TransformerBlock(
        embed_dim=64,  # 特征维度 (必须等于输入通道数C)
        num_heads=4,  # 注意力头数
        mlp_ratio=4.0,  # FFN隐藏层扩展比例
        dropout=0.1  # dropout率
    ).cuda()

    # 输入: (B, C, H, W)
    input_tensor = torch.randn(4, 64, 64, 64).cuda()
    output = transformer(input_tensor)  # 输出: (4, 256, 32, 32)
    print(output.shape)