"""
ViT推理脚本
简洁实用，直接加载checkpoint在测试集上运行
"""
import os
import sys
import yaml
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.vit import build_vit
from utils.data_loader import create_data_loaders, get_vit_transforms, FireDetectionDataset
from utils.metrics import calculate_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: str, model: nn.Module, device: torch.device) -> Dict[str, Any]:
    """加载checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    if 'metrics' in checkpoint:
        logger.info(f"Checkpoint metrics: {checkpoint['metrics']}")
    return checkpoint


def run_inference(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    classes: List[str],
) -> tuple:
    """在测试集上运行推理"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_image_paths = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Inference')
        for images, labels, image_paths in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_image_paths.extend(image_paths)
            all_probs.extend(probs.cpu().numpy())
    
    return all_predictions, all_labels, all_image_paths, all_probs


def save_results(
    predictions: np.ndarray,
    labels: np.ndarray,
    image_paths: List[str],
    probs: np.ndarray,
    classes: List[str],
    metrics: Dict[str, float],
    result_dir: Path,
    checkpoint_name: str = "best",
):
    """保存结果到文件"""
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存详细结果
    results = []
    for i, (pred, label, img_path, prob) in enumerate(zip(predictions, labels, image_paths, probs)):
        # 转换为Python原生类型，确保JSON可序列化
        pred = int(pred)
        label = int(label)
        results.append({
            "image_path": str(img_path),
            "true_label": classes[label],
            "predicted_label": classes[pred],
            "correct": bool(pred == label),
            "probabilities": {
                classes[j]: float(prob[j]) for j in range(len(classes))
            }
        })
    
    results_file = result_dir / f"{checkpoint_name}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved detailed results to {results_file}")
    
    # 保存指标
    metrics_file = result_dir / f"{checkpoint_name}_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_file}")
    
    # 保存混淆矩阵
    cm = confusion_matrix(labels, predictions)
    cm_file = result_dir / f"{checkpoint_name}_confusion_matrix.txt"
    with open(cm_file, 'w') as f:
        f.write("Confusion Matrix:\n")
        f.write("Rows: True labels, Columns: Predicted labels\n\n")
        f.write("Classes: " + ", ".join(classes) + "\n\n")
        f.write("     " + " ".join([f"{c:>8}" for c in classes]) + "\n")
        for i, class_name in enumerate(classes):
            f.write(f"{class_name:>4} " + " ".join([f"{cm[i,j]:>8}" for j in range(len(classes))]) + "\n")
    logger.info(f"Saved confusion matrix to {cm_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ViT Inference Script')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint file (default: best.pth from config)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (default: configs/vit/config.yaml)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: results/vit from config)'
    )
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = project_root / 'configs' / 'vit' / 'config.yaml'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 确定checkpoint路径
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_dir = project_root / config['paths']['checkpoint_dir']
        checkpoint_path = checkpoint_dir / 'best.pth'
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # 确定输出目录
    if args.output_dir:
        result_dir = Path(args.output_dir)
    else:
        result_dir = project_root / config['paths']['result_dir']
    
    checkpoint_name = checkpoint_path.stem
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Results will be saved to: {result_dir}")
    
    # 准备数据
    classes = ['no_fire', 'start_fire', 'fire']
    root_dir = project_root / config['data']['root_dir']
    test_dir = root_dir / 'test'
    
    # 检查是否有独立的测试集目录
    if test_dir.exists() and test_dir.is_dir():
        # 使用独立的测试集目录
        logger.info(f"Using dedicated test set from: {test_dir}")
        
        # 准备数据增强配置（测试时不需要增强）
        augmentation_config = config['data'].get('augmentation', {})
        if isinstance(augmentation_config, dict):
            augmentation_config = augmentation_config.copy()
            augmentation_config['enabled'] = False  # 测试时禁用增强
            augmentation_config['use_vit_transforms'] = True
        
        # 获取测试时的transforms
        test_transform = get_vit_transforms("test", int(config['model']['image_size']), augmentation_config)
        
        # 创建测试数据集（直接从test目录加载）
        test_dataset = FireDetectionDataset(
            root_dir=str(root_dir),
            databases=['test'],  # 使用test目录
            classes=classes,
            split='test',
            transform=test_transform,
            image_size=int(config['model']['image_size']),
        )
        
        # 创建测试数据加载器
        test_loader = DataLoader(
            test_dataset,
            batch_size=int(config['data']['batch_size']),
            shuffle=False,
            num_workers=int(config['data']['num_workers']),
            pin_memory=True,
        )
    # 创建模型
    model = build_vit(
        model_name=config['model']['name'],
        num_classes=int(config['model']['num_classes']),
        img_size=int(config['model']['image_size']),
        pretrained=bool(config['model'].get('pretrained', False)),
    )
    model = model.to(device)
    
    # 加载checkpoint
    load_checkpoint(str(checkpoint_path), model, device)
    
    # 运行推理
    logger.info("Running inference on test set...")
    predictions, labels, image_paths, probs = run_inference(
        model, test_loader, device, classes
    )
    
    # 计算指标
    metrics = calculate_metrics(predictions, labels, classes)
    
    # 打印指标
    logger.info("=" * 50)
    logger.info("Test Results:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    logger.info("=" * 50)
    
    # 保存结果
    save_results(
        np.array(predictions),
        np.array(labels),
        image_paths,
        np.array(probs),
        classes,
        metrics,
        result_dir,
        checkpoint_name,
    )
    
    logger.info("Inference completed!")


if __name__ == '__main__':
    main()

