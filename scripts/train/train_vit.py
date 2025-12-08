"""
Training script for Vision Transformer model
简洁实用的训练脚本，消除不必要的复杂性
"""
import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, skipping wandb logging")

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.vit import build_vit
from utils.data_loader import create_data_loaders
from utils.metrics import calculate_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (images, labels, _) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 更新进度条
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = running_loss / len(train_loader)
    metrics = calculate_metrics(all_labels, all_preds)
    metrics['loss'] = avg_loss
    
    return metrics


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """验证"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for images, labels, _ in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = running_loss / len(val_loader)
    metrics = calculate_metrics(all_labels, all_preds)
    metrics['loss'] = avg_loss
    
    return metrics


def get_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """创建优化器"""
    opt_name = config['training']['optimizer'].lower()
    # 确保learning_rate是浮点数
    lr = float(config['training']['learning_rate'])
    weight_decay = float(config['training'].get('weight_decay', 1e-4))
    
    if opt_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        momentum = float(config['training'].get('momentum', 0.9))
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def get_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any], num_epochs: int) -> optim.lr_scheduler._LRScheduler:
    """创建学习率调度器"""
    scheduler_name = config['training'].get('scheduler', 'cosine').lower()
    
    if scheduler_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_name == 'step':
        step_size = int(config['training'].get('step_size', 30))
        gamma = float(config['training'].get('gamma', 0.1))
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'none':
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    else:
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: Path,
    is_best: bool = False,
):
    """保存checkpoint"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
    }
    
    # 保存最新checkpoint
    checkpoint_path = checkpoint_dir / 'latest.pth'
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # 保存最佳checkpoint
    if is_best:
        best_path = checkpoint_dir / 'best.pth'
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best checkpoint to {best_path}")


def main():
    """主训练函数"""
    # 加载配置
    config_path = project_root / 'configs' / 'vit' / 'config.yaml'
    config = load_config(config_path)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 创建目录
    checkpoint_dir = project_root / config['paths']['checkpoint_dir']
    log_dir = project_root / config['paths']['log_dir']
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # 初始化wandb
    wandb_config = config.get('wandb', {})
    use_wandb = wandb_config.get('enabled', False) and WANDB_AVAILABLE
    
    if use_wandb:
        wandb.init(
            project=wandb_config.get('project', 'fire_detection'),
            name=wandb_config.get('name', None),
            config={
                'model': config['model'],
                'data': {
                    'batch_size': config['data']['batch_size'],
                    'image_size': config['model']['image_size'],
                    'train_split': config['data']['train_split'],
                    'val_split': config['data']['val_split'],
                    'test_split': config['data']['test_split'],
                },
                'training': {
                    'epochs': config['training']['epochs'],
                    'learning_rate': float(config['training']['learning_rate']),
                    'weight_decay': float(config['training'].get('weight_decay', 1e-4)),
                    'optimizer': config['training']['optimizer'],
                    'scheduler': config['training'].get('scheduler', 'cosine'),
                },
                'augmentation': config['data'].get('augmentation', {}),
            },
            dir=str(log_dir),
        )
        logger.info("Wandb initialized")
    
    # 准备数据增强配置（从配置文件读取，如果没有则使用默认值）
    if isinstance(config['data'].get('augmentation'), dict):
        augmentation_config = config['data']['augmentation'].copy()
        # 确保use_vit_transforms为True
        augmentation_config['use_vit_transforms'] = True
    else:
        # 默认数据增强配置
        augmentation_config = {
            'enabled': config['data'].get('augmentation', True),
            'use_vit_transforms': True,
            'horizontal_flip': 0.5,
            'color_jitter': True,
            'color_jitter_params': {
                'brightness': 0.2,
                'contrast': 0.2,
                'saturation': 0.2,
                'hue': 0.1,
                'p': 0.5,
            },
            'random_brightness_contrast': True,
            'brightness_limit': 0.2,
            'contrast_limit': 0.2,
            'brightness_contrast_p': 0.5,
            'rotate': True,
            'rotate_limit': 15,
            'rotate_p': 0.5,
            'shift_scale_rotate': True,
            'shift_limit': 0.1,
            'scale_limit': 0.1,
            'shift_scale_rotate_p': 0.5,
            'gaussian_noise': True,
            'noise_var_limit': (10.0, 50.0),
            'noise_p': 0.3,
            'gaussian_blur': True,
            'blur_limit': (3, 7),
            'blur_p': 0.3,
            'cutout': True,
            'max_holes': 8,
            'max_height': 32,
            'max_width': 32,
            'cutout_p': 0.3,
            'normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
            },
        }
    
    # 创建数据加载器
    classes = ['no_fire', 'start_fire', 'fire']
    databases = ['FIRE_DATABASE_1', 'FIRE_DATABASE_2', 'FIRE_DATABASE_3']
    
    train_loader, val_loader, test_loader = create_data_loaders(
        root_dir=str(project_root / config['data']['root_dir']),
        databases=databases,
        classes=classes,
        train_split=float(config['data']['train_split']),
        val_split=float(config['data']['val_split']),
        test_split=float(config['data']['test_split']),
        batch_size=int(config['data']['batch_size']),
        num_workers=int(config['data']['num_workers']),
        image_size=int(config['model']['image_size']),
        augmentation=augmentation_config,
    )
    
    # 创建模型
    model = build_vit(
        model_name=config['model']['name'],
        num_classes=int(config['model']['num_classes']),
        img_size=int(config['model']['image_size']),
        pretrained=bool(config['model'].get('pretrained', False)),
    )
    model = model.to(device)
    
    # 记录模型结构到wandb
    if use_wandb:
        # 计算模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.config.update({
            'model/total_params': total_params,
            'model/trainable_params': trainable_params,
        })
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = get_optimizer(model, config)
    
    # 学习率调度器
    num_epochs = int(config['training']['epochs'])
    scheduler = get_scheduler(optimizer, config, num_epochs)
    
    # 训练循环
    best_val_acc = 0.0
    start_epoch = 0
    
    logger.info("Starting training...")
    for epoch in range(start_epoch, num_epochs):
        # 训练
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, device, epoch)
        
        # 更新学习率
        scheduler.step()
        
        # 记录到TensorBoard
        writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
        writer.add_scalar('Train/Accuracy', train_metrics['accuracy'], epoch)
        writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
        writer.add_scalar('Val/Accuracy', val_metrics['accuracy'], epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # 记录到wandb
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'train/accuracy': train_metrics['accuracy'],
                'train/precision': train_metrics.get('precision', 0.0),
                'train/recall': train_metrics.get('recall', 0.0),
                'train/f1': train_metrics.get('f1', 0.0),
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'],
                'val/precision': val_metrics.get('precision', 0.0),
                'val/recall': val_metrics.get('recall', 0.0),
                'val/f1': val_metrics.get('f1', 0.0),
                'learning_rate': optimizer.param_groups[0]['lr'],
            }, step=epoch)
        
        # 打印指标
        logger.info(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f} - "
            f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}"
        )
        
        # 保存checkpoint
        is_best = val_metrics['accuracy'] > best_val_acc
        if is_best:
            best_val_acc = val_metrics['accuracy']
            # 记录最佳指标到wandb
            if use_wandb:
                wandb.run.summary['best_val_accuracy'] = best_val_acc
                wandb.run.summary['best_val_loss'] = val_metrics['loss']
                wandb.run.summary['best_epoch'] = epoch
        
        if (epoch + 1) % int(config['training'].get('save_interval', 5)) == 0 or is_best:
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, checkpoint_dir, is_best)
    
    writer.close()
    
    # 结束wandb
    if use_wandb:
        wandb.finish()
    
    logger.info("Training completed!")


if __name__ == '__main__':
    main()
