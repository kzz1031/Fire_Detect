"""
Data loading utilities for fire detection
"""
import os
import glob
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)


class FireDetectionDataset(Dataset):
    """
    Dataset for fire detection with three classes:
    - fire
    - start_fire
    - no_fire
    """
    
    def __init__(
        self,
        root_dir: str,
        databases: List[str],
        classes: List[str],
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        image_size: int = 448,
    ):
        """
        Initialize fire detection dataset.
        
        Args:
            root_dir: Root directory containing database folders
            databases: List of database folder names to use
            classes: List of class names
            split: Dataset split ('train', 'val', 'test')
            transform: Optional image transforms
            image_size: Target image size
        """
        self.root_dir = Path(root_dir)
        self.databases = databases
        self.classes = classes
        self.split = split
        self.image_size = image_size
        
        # Build class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Load image paths and labels
        self.samples = self._load_samples()
        
        # Set transforms
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
        
        logger.info(
            f"Loaded {len(self.samples)} samples for {split} split "
            f"from databases: {databases}"
        )
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load all image paths and their labels."""
        samples = []
        
        for db_name in self.databases:
            db_path = self.root_dir / db_name
            
            if not db_path.exists():
                logger.warning(f"Database {db_name} not found at {db_path}")
                continue
            
            for class_name in self.classes:
                class_path = db_path / class_name
                
                if not class_path.exists():
                    logger.warning(f"Class {class_name} not found in {db_name}")
                    continue
                
                # Find all image files
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
                image_paths = []
                for ext in image_extensions:
                    image_paths.extend(glob.glob(str(class_path / ext)))
                    image_paths.extend(glob.glob(str(class_path / ext.upper())))
                
                # Add to samples
                class_idx = self.class_to_idx[class_name]
                for img_path in image_paths:
                    samples.append((img_path, class_idx))
        
        return samples
    
    def _get_default_transform(self) -> transforms.Compose:
        """Get default image transforms."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, int, str]:
        """
        Get a sample from the dataset.
        
        Returns:
            image: PIL Image (for Qwen VLM) or transformed tensor
            label: Class index
            image_path: Path to the image file
        """
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
            # Resize if needed (before transforms for PIL)
            if image.size != (self.image_size, self.image_size):
                image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (self.image_size, self.image_size))
        
        # Apply transforms if provided (for ViT, transforms to tensor)
        # For Qwen VLM, we'll keep as PIL Image
        if self.transform:
            image = self.transform(image)
        
        return image, label, str(img_path)


def get_transforms(
    split: str = "train",
    image_size: int = 448,
    augmentation: dict = None,
    return_pil: bool = True
) -> Optional[transforms.Compose]:
    """
    Get data transforms for training or validation.
    
    Args:
        split: 'train' or 'val'/'test'
        image_size: Target image size
        augmentation: Augmentation configuration dict
        return_pil: If True, return None (keep PIL Image for Qwen VLM)
    
    Returns:
        transforms.Compose or None: Image transforms (None for Qwen VLM)
    """
    # For Qwen VLM, we keep images as PIL Images
    if return_pil:
        return None
    
    if split == "train" and augmentation and augmentation.get("enabled", False):
        # Training transforms with augmentation
        transform_list = [
            transforms.Resize((image_size, image_size)),
        ]
        
        if augmentation.get("random_crop", False):
            transform_list.append(transforms.RandomCrop(image_size))
        
        if augmentation.get("horizontal_flip", 0) > 0:
            transform_list.append(
                transforms.RandomHorizontalFlip(
                    p=augmentation["horizontal_flip"]
                )
            )
        
        if "color_jitter" in augmentation:
            cj = augmentation["color_jitter"]
            transform_list.append(
                transforms.ColorJitter(
                    brightness=cj.get("brightness", 0),
                    contrast=cj.get("contrast", 0),
                    saturation=cj.get("saturation", 0),
                    hue=cj.get("hue", 0),
                )
            )
        
        transform_list.extend([
            transforms.ToTensor(),
        ])
        
        if "normalize" in augmentation:
            norm = augmentation["normalize"]
            transform_list.append(
                transforms.Normalize(
                    mean=norm.get("mean", [0.485, 0.456, 0.406]),
                    std=norm.get("std", [0.229, 0.224, 0.225])
                )
            )
        else:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        return transforms.Compose(transform_list)
    else:
        # Validation/test transforms (no augmentation)
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def create_data_loaders(
    root_dir: str,
    databases: List[str],
    classes: List[str],
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True,
    image_size: int = 448,
    augmentation: dict = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        root_dir: Root directory containing database folders
        databases: List of database folder names
        classes: List of class names
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        test_split: Proportion of data for testing
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        image_size: Target image size
        augmentation: Augmentation configuration
        seed: Random seed for reproducibility
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create full dataset
    full_dataset = FireDetectionDataset(
        root_dir=root_dir,
        databases=databases,
        classes=classes,
        split="all",
        image_size=image_size,
    )
    
    # Split dataset
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size]
    )
    
    # Update split attribute for each subset
    train_dataset.dataset.split = "train"
    val_dataset.dataset.split = "val"
    test_dataset.dataset.split = "test"
    
    # Get transforms (None for Qwen VLM to keep PIL Images)
    train_transform = get_transforms("train", image_size, augmentation, return_pil=True)
    val_transform = get_transforms("val", image_size, augmentation, return_pil=True)
    test_transform = get_transforms("test", image_size, augmentation, return_pil=True)
    
    # Update transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = test_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    logger.info(
        f"Created data loaders - Train: {len(train_dataset)}, "
        f"Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )
    
    return train_loader, val_loader, test_loader
