"""
Vision Transformer (ViT) model for fire detection
简洁实用的实现，消除不必要的复杂性
"""
import torch
import torch.nn as nn
import math
from typing import Optional


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample - 随机丢弃路径以增强泛化"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # 生成随机tensor，形状为 [B, 1, 1] 以支持广播
        random_tensor = keep_prob + torch.rand(x.shape[0], 1, 1, device=x.device, dtype=x.dtype)
        random_tensor.floor_()  # 二值化：0或1
        output = x.div(keep_prob) * random_tensor
        return output


class PatchEmbedding(nn.Module):
    """将图像分割成patches并嵌入"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, embed_dim, n_patches_h, n_patches_w]
        x = self.proj(x)
        # Flatten: [B, embed_dim, n_patches_h, n_patches_w] -> [B, embed_dim, n_patches]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, n_patches, embed_dim]
        return x


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # 生成Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # 输出
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """前馈网络"""
    
    def __init__(self, embed_dim: int = 768, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer块：注意力 + MLP"""
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, mlp_ratio: int = 4, 
                 dropout: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer模型
    简洁实现，消除不必要的复杂性
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks - 使用线性衰减的drop path
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, dpr[i])
            for i in range(depth)
        ])
        
        # 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights_fn)
    
    def _init_weights_fn(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, n_patches, embed_dim]
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, n_patches+1, embed_dim]
        
        # 添加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # 分类
        x = self.norm(x)
        cls_token_final = x[:, 0]  # 使用CLS token
        logits = self.head(cls_token_final)
        
        return logits


def build_vit(
    model_name: str = "vit_base",
    num_classes: int = 3,
    img_size: int = 224,
    pretrained: bool = False,
    dropout: float = 0.0,
    drop_path: float = 0.0,
) -> VisionTransformer:
    """
    构建ViT模型
    
    Args:
        model_name: 模型名称 ('vit_tiny', 'vit_small', 'vit_base')
        num_classes: 分类数量
        img_size: 图像尺寸
        pretrained: 是否使用预训练权重（需要从timm加载）
        dropout: Dropout比率，用于减少过拟合
        drop_path: Drop path比率，用于正则化
    
    Returns:
        VisionTransformer模型
    """
    configs = {
        "vit_tiny": {
            "embed_dim": 192,
            "depth": 12,
            "num_heads": 3,
        },
        "vit_small": {
            "embed_dim": 384,
            "depth": 12,
            "num_heads": 6,
        },
        "vit_base": {
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
        },
    }
    
    if model_name not in configs:
        raise ValueError(f"Unknown model name: {model_name}. Choose from {list(configs.keys())}")
    
    config = configs[model_name]
    model = VisionTransformer(
        img_size=img_size,
        num_classes=num_classes,
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        dropout=dropout,
        drop_path=drop_path,
    )
    
    # 如果使用预训练权重，从timm加载
    if pretrained:
        try:
            import timm
            timm_model = timm.create_model(
                f"{model_name}_patch16_224",
                pretrained=True,
                num_classes=num_classes,
            )
            # 这里可以添加权重迁移逻辑
            # 为了简洁，暂时跳过
        except ImportError:
            print("Warning: timm not available, skipping pretrained weights")
    
    return model

