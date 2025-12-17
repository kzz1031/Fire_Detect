#!/usr/bin/env python3
"""
Visualize attention maps for ViT model.
支持单张图片的attention map可视化
"""
import sys
import yaml
import argparse
import logging
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.vit import build_vit
from utils.data_loader import get_vit_transforms

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)




def visualize_attention(
    model: nn.Module,
    image_path: str,
    transform,
    device: torch.device,
    layer_idx: Optional[int] = None,
    head_idx: Optional[int] = None,
    save_path: Optional[str] = None,
):
    """
    可视化单张图片的attention map
    
    Args:
        model: ViT模型
        image_path: 图片路径
        transform: 图像预处理
        device: 设备
        layer_idx: 指定层索引（None表示最后一层）
        head_idx: 指定head索引（None表示平均所有heads）
        save_path: 保存路径
    """
    # 加载图片
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    original_size = original_image.shape[:2]  # (H, W)
    
    # 预处理（注意：transform可能会改变图像尺寸）
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 修改每个TransformerBlock来捕获attention
    model.eval()
    attention_weights_list = []
    original_forwards = {}
    
    # 保存原始forward并创建新的forward
    for block in model.blocks:
        original_forwards[id(block.attn)] = block.attn.forward
        
        def make_forward(attn_module):
            def modified_forward(x):
                B, N, C = x.shape
                qkv = attn_module.qkv(x).reshape(B, N, 3, attn_module.num_heads, attn_module.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                attn = (q @ k.transpose(-2, -1)) * (attn_module.head_dim ** -0.5)
                attn = attn.softmax(dim=-1)
                attn_weights = attn.clone()
                attn = attn_module.dropout(attn)
                x_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x_out = attn_module.proj(x_out)
                return x_out, attn_weights
            return modified_forward
        
        block.attn.forward = make_forward(block.attn)
    
    # 获取attention weights
    with torch.no_grad():
        B = input_tensor.shape[0]
        
        # Patch embedding
        x = model.patch_embed(input_tensor)
        cls_tokens = model.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + model.pos_embed
        x = model.pos_drop(x)
        
        # 遍历每个block并获取attention
        for i, block in enumerate(model.blocks):
            x_norm = block.norm1(x)
            x_attn, attn_weights = block.attn.forward(x_norm)
            attention_weights_list.append(attn_weights.cpu())
            x = x + block.drop_path(x_attn)
            x = x + block.drop_path(block.mlp(block.norm2(x)))
    
    # 恢复原始forward方法
    for block in model.blocks:
        block.attn.forward = original_forwards[id(block.attn)]
    
    # 获取patch数量
    img_size = model.patch_embed.img_size
    patch_size = model.patch_embed.patch_size
    num_patches_h = img_size // patch_size
    num_patches_w = img_size // patch_size
    model_img_size = model.patch_embed.img_size
    
    def process_attention_weights(attn_weights_tensor, layer_idx):
        """处理单个层的attention weights"""
        attn_weights = attn_weights_tensor[0]  # [num_heads, N, N]
        
        # 选择head
        if head_idx is None:
            # 平均所有heads
            attn_weights = attn_weights.mean(dim=0)  # [N, N]
        else:
            attn_weights = attn_weights[head_idx]  # [N, N]
        
        # 提取CLS token的attention（第一行，去掉CLS token本身）
        cls_attention = attn_weights[0, 1:]  # [N-1] 去掉CLS token
        
        # Reshape attention到2D
        cls_attention_2d = cls_attention.reshape(num_patches_h, num_patches_w).numpy()
        
        # 上采样到模型输入尺寸
        try:
            from scipy.ndimage import zoom
            zoom_factor = model_img_size / num_patches_h
            attention_map_model_size = zoom(cls_attention_2d, (zoom_factor, zoom_factor), order=1)
        except ImportError:
            # 如果没有scipy，使用PIL resize
            from PIL import Image as PILImage
            attention_pil = PILImage.fromarray((cls_attention_2d * 255).astype(np.uint8))
            attention_pil = attention_pil.resize((model_img_size, model_img_size), PILImage.BILINEAR)
            attention_map_model_size = np.array(attention_pil).astype(float) / 255.0
        
        # 如果原图尺寸与模型输入尺寸不同，需要进一步resize到原图尺寸
        if original_size != (model_img_size, model_img_size):
            from PIL import Image as PILImage
            attention_pil = PILImage.fromarray((attention_map_model_size * 255).astype(np.uint8))
            attention_pil = attention_pil.resize((original_size[1], original_size[0]), PILImage.BILINEAR)
            attention_map = np.array(attention_pil).astype(float) / 255.0
        else:
            attention_map = attention_map_model_size
        
        return attention_map
    
    # 决定可视化哪些层
    if layer_idx is None:
        # 可视化所有层
        layers_to_visualize = list(range(len(attention_weights_list)))
        logger.info(f"Visualizing all {len(layers_to_visualize)} layers")
    else:
        layers_to_visualize = [layer_idx]
    
    # 处理所有层的attention
    attention_maps = []
    for idx in layers_to_visualize:
        attention_map = process_attention_weights(attention_weights_list[idx], idx)
        attention_maps.append(attention_map)
    
    # 可视化
    num_layers = len(layers_to_visualize)
    
    if num_layers == 1:
        # 单层：显示原图、attention map和叠加图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        layer_idx = layers_to_visualize[0]
        attention_map = attention_maps[0]
        
        # 原图
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')
        
        # Attention map
        im = axes[1].imshow(attention_map, cmap='hot', interpolation='bilinear')
        axes[1].set_title(f'Attention Map (Layer {layer_idx})', fontsize=14)
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # 叠加图
        axes[2].imshow(original_image)
        axes[2].imshow(attention_map, cmap='hot', alpha=0.5, interpolation='bilinear')
        axes[2].set_title('Overlay', fontsize=14)
        axes[2].axis('off')
    else:
        # 多层：网格布局显示所有层
        # 计算网格大小（尽量接近正方形）
        cols = 4
        rows = (num_layers + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, (layer_idx, attention_map) in enumerate(zip(layers_to_visualize, attention_maps)):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # 显示叠加图
            ax.imshow(original_image)
            im = ax.imshow(attention_map, cmap='hot', alpha=0.5, interpolation='bilinear')
            ax.set_title(f'Layer {layer_idx}', fontsize=12)
            ax.axis('off')
        
        # 隐藏多余的子图
        for idx in range(num_layers, rows * cols):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved attention visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Visualize ViT attention maps')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint file'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (default: configs/vit/config.yaml)'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to image file to visualize'
    )
    parser.add_argument(
        '--layer',
        type=int,
        default=None,
        help='Layer index to visualize (default: all layers)'
    )
    parser.add_argument(
        '--all-layers',
        action='store_true',
        help='Visualize all layers (same as --layer None)'
    )
    parser.add_argument(
        '--head',
        type=int,
        default=None,
        help='Head index to visualize (default: average all heads)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for visualization (default: show interactively)'
    )
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = project_root / 'configs' / 'vit' / 'config.yaml'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Check checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Checkpoint: {checkpoint_path}")
    
    # Check image
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    logger.info(f"Image: {image_path}")
    
    # Prepare transforms
    augmentation_config = config['data'].get('augmentation', {})
    if isinstance(augmentation_config, dict):
        augmentation_config = augmentation_config.copy()
        augmentation_config['enabled'] = False
        augmentation_config['use_vit_transforms'] = True
    
    transform = get_vit_transforms(
        "test",
        int(config['model']['image_size']),
        augmentation_config
    )
    
    # Create model
    model = build_vit(
        model_name=config['model']['name'],
        num_classes=int(config['model']['num_classes']),
        img_size=int(config['model']['image_size']),
        pretrained=bool(config['model'].get('pretrained', False)),
        dropout=float(config['model'].get('dropout', 0.0)),
        drop_path=float(config['model'].get('drop_path', 0.0)),
    )
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    # Determine layer to visualize
    layer_idx = None if args.all_layers else args.layer
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        if layer_idx is None:
            output_path = f"attention_{image_path.stem}_all_layers.png"
        else:
            output_path = f"attention_{image_path.stem}_layer{layer_idx}.png"
    
    # Visualize
    logger.info("Generating attention visualization...")
    visualize_attention(
        model=model,
        image_path=str(image_path),
        transform=transform,
        device=device,
        layer_idx=layer_idx,
        head_idx=args.head,
        save_path=output_path,
    )
    
    logger.info("Visualization completed!")


if __name__ == '__main__':
    main()
