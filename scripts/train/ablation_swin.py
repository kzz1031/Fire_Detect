import os
import sys
import yaml
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, skipping wandb logging")

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.swin import build_swin
from utils.data_loader import create_data_loaders
from utils.metrics import calculate_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance and hard examples"""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        import torch.nn.functional as F
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config_to_checkpoint(config_path: Path, checkpoint_dir: Path, exp_name: str):
    """保存配置文件到checkpoint目录，带实验名称"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 生成新的文件名：config_EXPNAME_TIMESTAMP.yaml
    config_filename = f"config_{exp_name}_{timestamp}.yaml"
    config_dest = checkpoint_dir / config_filename
    
    # 复制配置文件
    shutil.copy2(config_path, config_dest)
    logger.info(f"Saved config to {config_dest}")
    
    # 同时保存一个latest.yaml方便查看
    latest_config = checkpoint_dir / "config_latest.yaml"
    shutil.copy2(config_path, latest_config)
    logger.info(f"Saved latest config to {latest_config}")


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
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
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


def get_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any], num_epochs: int):
    """创建学习率调度器"""
    import math
    scheduler_name = config['training'].get('scheduler', 'cosine').lower()
    warmup_epochs = int(config['training'].get('warmup_epochs', 0))
    
    if warmup_epochs > 0:
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                base_epoch = epoch - warmup_epochs
                if scheduler_name == 'cosine':
                    T_max = num_epochs - warmup_epochs
                    return 0.5 * (1 + math.cos(math.pi * base_epoch / T_max))
                elif scheduler_name == 'step':
                    step_size = int(config['training'].get('step_size', 30))
                    gamma = float(config['training'].get('gamma', 0.1))
                    return gamma ** (base_epoch // step_size)
                else:
                    return 1.0
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return scheduler
    
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
    
    checkpoint_path = checkpoint_dir / 'latest.pth'
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    if is_best:
        best_path = checkpoint_dir / 'best.pth'
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best checkpoint to {best_path}")


def run_experiment(
    exp_name: str,
    config: Dict[str, Any],
    base_config_path: Path,
    device: torch.device,
):
    """运行单个实验"""
    logger.info("=" * 80)
    logger.info(f"Starting experiment: {exp_name}")
    logger.info("=" * 80)
    
    # 创建实验专用的目录
    checkpoint_dir = project_root / 'checkpoints' / 'swin' / exp_name
    log_dir = project_root / 'logs' / 'swin' / exp_name
    result_dir = project_root / 'results' / 'swin' / exp_name
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # 初始化wandb（可选）
    wandb_config = config.get('wandb', {})
    use_wandb = wandb_config.get('enabled', False) and WANDB_AVAILABLE
    
    if use_wandb:
        wandb.init(
            project=wandb_config.get('project', 'fire_detection_swin_ablation'),
            name=exp_name,
            config=config,
            dir=str(log_dir),
            reinit=True,
        )
        logger.info("Wandb initialized")
    
    # 准备数据增强配置
    if isinstance(config['data'].get('augmentation'), dict):
        augmentation_config = config['data']['augmentation'].copy()
        augmentation_config['use_vit_transforms'] = True
    else:
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
    model = build_swin(
        model_name=config['model']['name'],
        num_classes=int(config['model']['num_classes']),
        img_size=int(config['model']['image_size']),
        pretrained=bool(config['model'].get('pretrained', False)),
        drop_rate=float(config['model'].get('drop_rate', 0.0)),
        drop_path_rate=float(config['model'].get('drop_path_rate', 0.1)),
    )
    model = model.to(device)
    
    # 记录模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    if use_wandb:
        wandb.config.update({
            'model/total_params': total_params,
            'model/trainable_params': trainable_params,
        })
    
    # 损失函数
    training_config = config['training']
    use_focal_loss = training_config.get('use_focal_loss', False)
    use_class_weights = training_config.get('use_class_weights', False)
    label_smoothing = training_config.get('label_smoothing', 0.0)
    
    if use_focal_loss:
        focal_alpha = float(training_config.get('focal_loss_alpha', 1.0))
        focal_gamma = float(training_config.get('focal_loss_gamma', 2.0))
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        logger.info(f"Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
    elif use_class_weights:
        all_labels = []
        for _, labels, _ in train_loader:
            all_labels.extend(labels.numpy())
        all_labels = np.array(all_labels)
        
        class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        logger.info(f"Using weighted CrossEntropyLoss with class weights: {class_weights.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        if label_smoothing > 0:
            logger.info(f"Using CrossEntropyLoss with label smoothing: {label_smoothing}")
        else:
            logger.info("Using standard CrossEntropyLoss")
    
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
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_metrics = validate(model, val_loader, criterion, device, epoch)
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
            if use_wandb:
                wandb.run.summary['best_val_accuracy'] = best_val_acc
                wandb.run.summary['best_val_loss'] = val_metrics['loss']
                wandb.run.summary['best_epoch'] = epoch
        
        if (epoch + 1) % int(config['training'].get('save_interval', 5)) == 0 or is_best:
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, checkpoint_dir, is_best)
    
    writer.close()
    
    if use_wandb:
        wandb.finish()
    
    # 保存配置文件
    save_config_to_checkpoint(base_config_path, checkpoint_dir, exp_name)
    
    logger.info(f"Experiment {exp_name} completed! Best val accuracy: {best_val_acc:.4f}")
    logger.info("=" * 80)
    
    return best_val_acc


def main():
    """主函数：运行消融实验"""
    # 加载基础配置
    base_config_path = project_root / 'configs' / 'swin' / 'config.yaml'
    base_config = load_config(str(base_config_path))
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 定义消融实验配置
    ablation_experiments = [
        # 实验1: 不同模型大小
        {
            'name': 'swin_tiny',
            'config_updates': {
                'model': {'name': 'swin_tiny'},
                'paths': {
                    'checkpoint_dir': 'checkpoints/swin/swin_tiny',
                    'log_dir': 'logs/swin/swin_tiny',
                    'result_dir': 'results/swin/swin_tiny',
                }
            }
        },
        {
            'name': 'swin_small',
            'config_updates': {
                'model': {'name': 'swin_small'},
                'paths': {
                    'checkpoint_dir': 'checkpoints/swin/swin_small',
                    'log_dir': 'logs/swin/swin_small',
                    'result_dir': 'results/swin/swin_small',
                }
            }
        },
        {
            'name': 'swin_base',
            'config_updates': {
                'model': {'name': 'swin_base'},
                'paths': {
                    'checkpoint_dir': 'checkpoints/swin/swin_base',
                    'log_dir': 'logs/swin/swin_base',
                    'result_dir': 'results/swin/swin_base',
                }
            }
        },
        
        # 实验2: 不同学习率
        {
            'name': 'lr_5e-5',
            'config_updates': {
                'training': {'learning_rate': 5e-5},
                'paths': {
                    'checkpoint_dir': 'checkpoints/swin/lr_5e-5',
                    'log_dir': 'logs/swin/lr_5e-5',
                    'result_dir': 'results/swin/lr_5e-5',
                }
            }
        },
        {
            'name': 'lr_1e-4',
            'config_updates': {
                'training': {'learning_rate': 1e-4},
                'paths': {
                    'checkpoint_dir': 'checkpoints/swin/lr_1e-4',
                    'log_dir': 'logs/swin/lr_1e-4',
                    'result_dir': 'results/swin/lr_1e-4',
                }
            }
        },
        {
            'name': 'lr_2e-4',
            'config_updates': {
                'training': {'learning_rate': 2e-4},
                'paths': {
                    'checkpoint_dir': 'checkpoints/swin/lr_2e-4',
                    'log_dir': 'logs/swin/lr_2e-4',
                    'result_dir': 'results/swin/lr_2e-4',
                }
            }
        },
        
        # 实验3: 不同dropout率
        {
            'name': 'dropout_0.0',
            'config_updates': {
                'model': {'drop_rate': 0.0, 'drop_path_rate': 0.0},
                'paths': {
                    'checkpoint_dir': 'checkpoints/swin/dropout_0.0',
                    'log_dir': 'logs/swin/dropout_0.0',
                    'result_dir': 'results/swin/dropout_0.0',
                }
            }
        },
        {
            'name': 'dropout_0.1',
            'config_updates': {
                'model': {'drop_rate': 0.1, 'drop_path_rate': 0.1},
                'paths': {
                    'checkpoint_dir': 'checkpoints/swin/dropout_0.1',
                    'log_dir': 'logs/swin/dropout_0.1',
                    'result_dir': 'results/swin/dropout_0.1',
                }
            }
        },
        {
            'name': 'dropout_0.2',
            'config_updates': {
                'model': {'drop_rate': 0.2, 'drop_path_rate': 0.2},
                'paths': {
                    'checkpoint_dir': 'checkpoints/swin/dropout_0.2',
                    'log_dir': 'logs/swin/dropout_0.2',
                    'result_dir': 'results/swin/dropout_0.2',
                }
            }
        },
        
        # 实验4: 不同batch size
        {
            'name': 'bs_32',
            'config_updates': {
                'data': {'batch_size': 32},
                'paths': {
                    'checkpoint_dir': 'checkpoints/swin/bs_32',
                    'log_dir': 'logs/swin/bs_32',
                    'result_dir': 'results/swin/bs_32',
                }
            }
        },
        {
            'name': 'bs_64',
            'config_updates': {
                'data': {'batch_size': 64},
                'paths': {
                    'checkpoint_dir': 'checkpoints/swin/bs_64',
                    'log_dir': 'logs/swin/bs_64',
                    'result_dir': 'results/swin/bs_64',
                }
            }
        },
        {
            'name': 'bs_128',
            'config_updates': {
                'data': {'batch_size': 128},
                'paths': {
                    'checkpoint_dir': 'checkpoints/swin/bs_128',
                    'log_dir': 'logs/swin/bs_128',
                    'result_dir': 'results/swin/bs_128',
                }
            }
        },
        
        # 实验5: 使用Focal Loss
        {
            'name': 'focal_loss',
            'config_updates': {
                'training': {
                    'use_focal_loss': True,
                    'focal_loss_alpha': 1.0,
                    'focal_loss_gamma': 2.0,
                },
                'paths': {
                    'checkpoint_dir': 'checkpoints/swin/focal_loss',
                    'log_dir': 'logs/swin/focal_loss',
                    'result_dir': 'results/swin/focal_loss',
                }
            }
        },
        
        # 实验6: 使用类别权重
        {
            'name': 'class_weights',
            'config_updates': {
                'training': {
                    'use_class_weights': True,
                },
                'paths': {
                    'checkpoint_dir': 'checkpoints/swin/class_weights',
                    'log_dir': 'logs/swin/class_weights',
                    'result_dir': 'results/swin/class_weights',
                }
            }
        },
    ]
    
    # 运行所有实验
    results = {}
    for exp in ablation_experiments:
        exp_name = exp['name']
        config_updates = exp['config_updates']
        
        # 创建实验配置（深拷贝基础配置并更新）
        import copy
        exp_config = copy.deepcopy(base_config)
        
        # 递归更新配置
        def update_dict(base, updates):
            for key, value in updates.items():
                if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                    update_dict(base[key], value)
                else:
                    base[key] = value
        
        update_dict(exp_config, config_updates)
        
        # 运行实验
        try:
            best_acc = run_experiment(exp_name, exp_config, base_config_path, device)
            results[exp_name] = best_acc
        except Exception as e:
            logger.error(f"Experiment {exp_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results[exp_name] = None
    
    # 打印所有实验结果
    logger.info("=" * 80)
    logger.info("Ablation Study Results Summary:")
    logger.info("=" * 80)
    for exp_name, best_acc in results.items():
        if best_acc is not None:
            logger.info(f"{exp_name:30s}: {best_acc:.4f}")
        else:
            logger.info(f"{exp_name:30s}: FAILED")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

