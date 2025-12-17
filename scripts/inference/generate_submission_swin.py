#!/usr/bin/env python3
"""
Generate submission CSV file for fire detection task.

CSV format:
- ID: Unique identifier of the image (filename with extension, e.g., "F_1000.jpg")
- Label: 0 for Fire, 1 for No Fire, 2 for Fire Start
"""
import os
import sys
import yaml
import csv
import argparse
import logging
import glob
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.swin import build_swin
from utils.data_loader import get_vit_transforms
import numpy as np

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestDataset(Dataset):
    """Simple dataset for test images without class labels."""
    
    def __init__(
        self,
        test_dir: Path,
        transform: Optional = None,
        image_size: int = 224,
    ):
        self.test_dir = Path(test_dir)
        self.transform = transform
        self.image_size = image_size
        
        # Load all image files from test directory
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
        self.image_paths = []
        for ext in image_extensions:
            self.image_paths.extend(glob.glob(str(self.test_dir / ext)))
        
        self.image_paths = sorted(self.image_paths)
        logger.info(f"Found {len(self.image_paths)} images in {test_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if image.size != (self.image_size, self.image_size):
                image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (self.image_size, self.image_size))
        
        if self.transform:
            # Handle both albumentations and torchvision transforms
            if ALBUMENTATIONS_AVAILABLE:
                try:
                    # Try albumentations (needs numpy array)
                    image_np = np.array(image)
                    result = self.transform(image=image_np)
                    if isinstance(result, dict):
                        image = result['image']
                    else:
                        image = result
                except (TypeError, KeyError, AttributeError):
                    # Fallback to torchvision (needs PIL Image)
                    image = self.transform(image)
            else:
                # torchvision transforms need PIL Image
                image = self.transform(image)
        
        return image, str(img_path)


def load_checkpoint(checkpoint_path: str, model: nn.Module, device: torch.device):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint


def predict_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> List[Tuple[str, int]]:
    """
    Run inference on test set and return predictions.
    
    Returns:
        List of (image_id, label) tuples where label is in submission format:
        0=Fire, 1=No Fire, 2=Fire Start
    """
    model.eval()
    results = []
    
    # Model class indices: 0=no_fire, 1=start_fire, 2=fire
    # Submission format: 0=Fire, 1=No Fire, 2=Fire Start
    # Mapping: no_fire(0)->1, start_fire(1)->2, fire(2)->0
    label_mapping = {0: 1, 1: 2, 2: 0}
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Generating predictions')
        for images, image_paths in pbar:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            for img_path, pred in zip(image_paths, preds.cpu().numpy()):
                # Extract ImageID from path (filename with extension)
                img_path = Path(img_path)
                image_id = img_path.name  # Include extension as required by Kaggle
                
                # Map model prediction to submission format
                submission_label = label_mapping[int(pred)]
                results.append((image_id, submission_label))
    
    return results


def save_submission_csv(results: List[Tuple[str, int]], output_path: Path):
    """Save predictions to CSV file in submission format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Label'])  # Kaggle expects 'ID' not 'ImageID'
        writer.writerows(results)
    
    logger.info(f"Saved submission file to {output_path}")
    logger.info(f"Total predictions: {len(results)}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate submission CSV file')
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
        help='Path to config file (default: configs/swin/config.yaml)'
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        default=None,
        help='Path to test directory (default: data/test from config)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='submission.csv',
        help='Output CSV file path (default: submission.csv)'
    )
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = project_root / 'configs' / 'swin' / 'config.yaml'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_dir = project_root / config['paths']['checkpoint_dir']
        checkpoint_path = checkpoint_dir / 'best.pth'
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Checkpoint: {checkpoint_path}")
    
    # Determine test directory
    if args.test_dir:
        test_dir = Path(args.test_dir)
    else:
        root_dir = project_root / config['data']['root_dir']
        test_dir = root_dir / 'test'
    
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    logger.info(f"Test directory: {test_dir}")
    
    # Prepare test dataset
    augmentation_config = config['data'].get('augmentation', {})
    if isinstance(augmentation_config, dict):
        augmentation_config = augmentation_config.copy()
        augmentation_config['enabled'] = False
        augmentation_config['use_vit_transforms'] = True
    
    test_transform = get_vit_transforms(
        "test",
        int(config['model']['image_size']),
        augmentation_config
    )
    
    # Create test dataset (load all images from test directory)
    test_dataset = TestDataset(
        test_dir=test_dir,
        transform=test_transform,
        image_size=int(config['model']['image_size']),
    )
    
    if len(test_dataset) == 0:
        raise ValueError(f"No test images found in {test_dir}")
    
    # Create test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(config['data']['batch_size']),
        shuffle=False,
        num_workers=int(config['data']['num_workers']),
        pin_memory=True,
    )
    
    # Create model
    model = build_swin(
        model_name=config['model']['name'],
        num_classes=int(config['model']['num_classes']),
        img_size=int(config['model']['image_size']),
        pretrained=bool(config['model'].get('pretrained', False)),
        drop_rate=float(config['model'].get('drop_rate', 0.0)),
        drop_path_rate=float(config['model'].get('drop_path_rate', 0.1)),
    )
    model = model.to(device)
    
    # Load checkpoint
    load_checkpoint(str(checkpoint_path), model, device)
    
    # Generate predictions
    logger.info("Running inference on test set...")
    results = predict_test_set(model, test_loader, device)
    
    # Save submission CSV
    output_path = Path(args.output)
    save_submission_csv(results, output_path)
    
    logger.info("Submission file generated successfully!")


if __name__ == '__main__':
    main()
