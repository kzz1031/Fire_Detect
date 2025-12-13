"""
Swin Transformer model for fire detection
简洁实用的实现，消除不必要的复杂性
基于分层窗口注意力机制，比ViT更高效
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    将特征图分割成窗口
    x: [B, H, W, C]
    return: [B*num_windows, window_size, window_size, C]
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    将窗口还原为特征图
    windows: [B*num_windows, window_size, window_size, C]
    return: [B, H, W, C]
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """基于窗口的多头自注意力"""
    
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # 相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        # 相对位置索引
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B*num_windows, window_size*window_size, C]
        mask: [num_windows, window_size*window_size, window_size*window_size] or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def create_mask(window_size: int, shift_size: int, H: int, W: int) -> torch.Tensor:
    """创建shifted window的mask"""
    img_mask = torch.zeros((1, H, W, 1))
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    
    return attn_mask


class SwinTransformerBlock(nn.Module):
    """Swin Transformer块"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int = 0,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size, num_heads, qkv_bias, attn_drop, drop
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, drop)
    
    def forward(self, x: torch.Tensor, H: int, W: int, mask_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, H*W, C]
        """
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # 循环填充
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        
        # 窗口分割
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # 窗口注意力
        attn_windows = self.attn(x_windows, mask=attn_mask)
        
        # 窗口还原
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        
        # 循环还原
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class PatchEmbedding(nn.Module):
    """Patch嵌入层"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 4, in_channels: int = 3, embed_dim: int = 96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        x: [B, C, H, W]
        return: [B, H*W/patch_size^2, embed_dim], H/patch_size, W/patch_size
        """
        B, C, H, W = x.shape
        x = self.proj(x)
        _, _, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, Hp, Wp


class PatchMerging(nn.Module):
    """Patch合并层，用于下采样"""
    
    def __init__(self, dim: int, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        """
        x: [B, H*W, C]
        return: [B, H*W/4, 2*C], H/2, W/2
        """
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        
        # 2x2合并
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        
        x = self.norm(x)
        x = self.reduction(x)
        
        return x, H // 2, W // 2


class MLP(nn.Module):
    """前馈网络"""
    
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Drop path (Stochastic Depth)"""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        random_tensor = keep_prob + torch.rand(x.shape[0], 1, 1, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class SwinTransformer(nn.Module):
    """
    Swin Transformer模型
    分层架构，窗口注意力，更高效
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 3,
        embed_dim: int = 96,
        depths: list = [2, 2, 6, 2],
        num_heads: list = [3, 6, 12, 24],
        window_size: int = 7,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        patches_resolution = self.patch_embed.img_size // self.patch_embed.patch_size
        self.patches_resolution = patches_resolution
        
        # 绝对位置编码（可选，Swin通常不需要）
        # self.pos_drop = nn.Dropout(p=drop_rate)
        
        # 构建stage
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = self._make_layer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
            )
            self.layers.append(layer)
        
        # 分类头
        self.norm = nn.LayerNorm(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        
        self._init_weights()
    
    def _make_layer(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: int,
        qkv_bias: bool,
        drop: float,
        attn_drop: float,
        drop_path: list,
        downsample: Optional[nn.Module] = None,
    ) -> nn.ModuleList:
        blocks = nn.ModuleList()
        for i in range(depth):
            shift_size = 0 if (i % 2 == 0) else window_size // 2
            blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                )
            )
        
        if downsample is not None:
            downsample_layer = downsample(dim=dim)
        else:
            downsample_layer = None
        
        return nn.ModuleList([blocks, downsample_layer])
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征"""
        x, H, W = self.patch_embed(x)
        # x = self.pos_drop(x)
        
        for layer in self.layers:
            blocks, downsample = layer
            for block in blocks:
                # 创建mask（仅在shifted window时需要）
                if block.shift_size > 0:
                    mask_matrix = create_mask(block.window_size, block.shift_size, H, W).to(x.device)
                else:
                    mask_matrix = None
                x = block(x, H, W, mask_matrix)
            
            if downsample is not None:
                x, H, W = downsample(x, H, W)
        
        x = self.norm(x)
        return x, H, W
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x, H, W = self.forward_features(x)
        x = x.transpose(1, 2)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.head(x)
        return x


def build_swin(
    model_name: str = "swin_tiny",
    num_classes: int = 3,
    img_size: int = 224,
    pretrained: bool = False,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
) -> SwinTransformer:
    """
    构建Swin Transformer模型
    
    Args:
        model_name: 模型名称 ('swin_tiny', 'swin_small', 'swin_base')
        num_classes: 分类数量
        img_size: 图像尺寸
        pretrained: 是否使用预训练权重
        drop_rate: Dropout比率
        drop_path_rate: Drop path比率
    
    Returns:
        SwinTransformer模型
    """
    configs = {
        "swin_tiny": {
            "embed_dim": 96,
            "depths": [2, 2, 6, 2],
            "num_heads": [3, 6, 12, 24],
        },
        "swin_small": {
            "embed_dim": 96,
            "depths": [2, 2, 18, 2],
            "num_heads": [3, 6, 12, 24],
        },
        "swin_base": {
            "embed_dim": 128,
            "depths": [2, 2, 18, 2],
            "num_heads": [4, 8, 16, 32],
        },
    }
    
    if model_name not in configs:
        raise ValueError(f"Unknown model name: {model_name}. Choose from {list(configs.keys())}")
    
    config = configs[model_name]
    model = SwinTransformer(
        img_size=img_size,
        num_classes=num_classes,
        embed_dim=config["embed_dim"],
        depths=config["depths"],
        num_heads=config["num_heads"],
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )
    
    if pretrained:
        # 可以在这里加载预训练权重
        pass
    
    return model

