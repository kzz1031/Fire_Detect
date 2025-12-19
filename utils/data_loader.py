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
import numpy as np

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations not available, using torchvision transforms only")

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
        
        # Apply transforms if provided
        # 支持albumentations (需要numpy数组) 和torchvision (需要PIL Image)
        if self.transform:
            # 检查是否是albumentations transform
            if ALBUMENTATIONS_AVAILABLE:
                try:
                    # 尝试作为albumentations transform (需要numpy数组)
                    image_np = np.array(image)
                    result = self.transform(image=image_np)
                    # albumentations返回字典，torchvision返回tensor
                    if isinstance(result, dict):
                        image = result['image']
                    else:
                        image = result
                except (TypeError, KeyError, AttributeError):
                    # 如果不是albumentations，使用PIL Image (torchvision)
                    image = self.transform(image)
            else:
                # torchvision transforms需要PIL Image
                image = self.transform(image)
        
        return image, label, str(img_path)


def get_vit_transforms(
    split: str = "train",
    image_size: int = 224,
    augmentation: dict = None,
) -> transforms.Compose:
    if augmentation:
        resize_strategy = augmentation.get("resize_strategy", "crop")
    else:
        resize_strategy = "crop"  # 默认使用crop策略保持长宽比
    
    if split == "train" and augmentation and augmentation.get("enabled", False):
        if ALBUMENTATIONS_AVAILABLE:
            # 使用albumentations进行数据增强
            transform_list = []
            
            # 根据策略选择resize方式
            if resize_strategy == "squash":
                # 强制拉伸（原始方式，不推荐）
                transform_list.append(A.Resize(image_size, image_size))
            elif resize_strategy == "crop":
                # 保持长宽比，短边resize到image_size（或稍大），然后crop
                # 这样高分辨率图片能保留更多信息
                # 训练时：短边resize到稍大尺寸，然后随机crop增加数据多样性
                resize_scale = augmentation.get("resize_scale", 1.1)  # 放大1.1倍再crop
                transform_list.append(
                    A.SmallestMaxSize(max_size=int(image_size * resize_scale))
                )
                transform_list.append(
                    A.RandomCrop(height=image_size, width=image_size)
                )
            elif resize_strategy == "pad":
                # 保持长宽比，短边resize到image_size，然后pad
                transform_list.append(
                    A.SmallestMaxSize(max_size=image_size)
                )
                transform_list.append(
                    A.PadIfNeeded(
                        min_height=image_size,
                        min_width=image_size,
                        border_mode=0,  # 使用0填充（黑色）
                        value=0,
                        mask_value=0
                    )
                )
            else:
                # 默认使用crop策略
                resize_scale = augmentation.get("resize_scale", 1.1)
                transform_list.append(
                    A.SmallestMaxSize(max_size=int(image_size * resize_scale))
                )
                transform_list.append(
                    A.RandomCrop(height=image_size, width=image_size)
                )
            
            # 几何变换
            if augmentation.get("horizontal_flip", 0) > 0:
                transform_list.append(
                    A.HorizontalFlip(p=augmentation["horizontal_flip"])
                )
            
            if augmentation.get("vertical_flip", False):
                transform_list.append(
                    A.VerticalFlip(p=augmentation.get("vertical_flip_p", 0.5))
                )
            
            if augmentation.get("rotate", False):
                transform_list.append(
                    A.Rotate(
                        limit=augmentation.get("rotate_limit", 15),
                        p=augmentation.get("rotate_p", 0.5)
                    )
                )
            
            if augmentation.get("shift_scale_rotate", False):
                transform_list.append(
                    A.ShiftScaleRotate(
                        shift_limit=augmentation.get("shift_limit", 0.1),
                        scale_limit=augmentation.get("scale_limit", 0.1),
                        rotate_limit=augmentation.get("rotate_limit", 10),
                        p=augmentation.get("shift_scale_rotate_p", 0.5)
                    )
                )
            
            # 颜色增强
            if augmentation.get("color_jitter", False):
                cj = augmentation.get("color_jitter_params", {})
                if not isinstance(cj, dict):
                    cj = {}
                transform_list.append(
                    A.ColorJitter(
                        brightness=cj.get("brightness", 0.2),
                        contrast=cj.get("contrast", 0.2),
                        saturation=cj.get("saturation", 0.2),
                        hue=cj.get("hue", 0.1),
                        p=cj.get("p", 0.5)
                    )
                )
            
            if augmentation.get("random_brightness_contrast", False):
                transform_list.append(
                    A.RandomBrightnessContrast(
                        brightness_limit=augmentation.get("brightness_limit", 0.2),
                        contrast_limit=augmentation.get("contrast_limit", 0.2),
                        p=augmentation.get("brightness_contrast_p", 0.5)
                    )
                )
            
            if augmentation.get("hue_saturation_value", False):
                transform_list.append(
                    A.HueSaturationValue(
                        hue_shift_limit=augmentation.get("hue_shift_limit", 20),
                        sat_shift_limit=augmentation.get("sat_shift_limit", 30),
                        val_shift_limit=augmentation.get("val_shift_limit", 20),
                        p=augmentation.get("hsv_p", 0.5)
                    )
                )
            
            # RGB通道独立调整，减少对单一颜色通道的依赖
            if augmentation.get("rgb_shift", False):
                rgb_shift_limit = augmentation.get("rgb_shift_limit", 20)
                transform_list.append(
                    A.RGBShift(
                        r_shift_limit=(-rgb_shift_limit, rgb_shift_limit),
                        g_shift_limit=(-rgb_shift_limit, rgb_shift_limit),
                        b_shift_limit=(-rgb_shift_limit, rgb_shift_limit),
                        p=augmentation.get("rgb_shift_p", 0.5)
                    )
                )
            
            # 噪声和模糊
            if augmentation.get("gaussian_noise", False):
                noise_var_limit = augmentation.get("noise_var_limit", (10.0, 50.0))
                # 如果是列表，转换为元组
                if isinstance(noise_var_limit, list):
                    noise_var_limit = tuple(noise_var_limit)
                transform_list.append(
                    A.GaussNoise(
                        var_limit=noise_var_limit,
                        p=augmentation.get("noise_p", 0.3)
                    )
                )
            
            if augmentation.get("gaussian_blur", False):
                blur_limit = augmentation.get("blur_limit", (3, 7))
                # 如果是列表，转换为元组
                if isinstance(blur_limit, list):
                    blur_limit = tuple(blur_limit)
                transform_list.append(
                    A.GaussianBlur(
                        blur_limit=blur_limit,
                        p=augmentation.get("blur_p", 0.3)
                    )
                )
            
            # 遮挡
            if augmentation.get("cutout", False):
                transform_list.append(
                    A.CoarseDropout(
                        max_holes=augmentation.get("max_holes", 8),
                        max_height=augmentation.get("max_height", 32),
                        max_width=augmentation.get("max_width", 32),
                        p=augmentation.get("cutout_p", 0.3)
                    )
                )
            
            # 归一化和转换为tensor
            norm_mean = augmentation.get("normalize", {}).get("mean", [0.485, 0.456, 0.406])
            norm_std = augmentation.get("normalize", {}).get("std", [0.229, 0.224, 0.225])
            
            transform_list.extend([
                A.Normalize(mean=norm_mean, std=norm_std),
                ToTensorV2(),
            ])
            
            return A.Compose(transform_list)
        else:
            # 回退到torchvision transforms（也支持保持长宽比）
            transform_list = []
            
            # 根据策略选择resize方式
            if resize_strategy == "squash":
                transform_list.append(transforms.Resize((image_size, image_size)))
            elif resize_strategy == "crop":
                # 训练时：短边resize到稍大尺寸，然后随机crop
                resize_scale = augmentation.get("resize_scale", 1.1)
                transform_list.append(transforms.Resize(int(image_size * resize_scale)))
                transform_list.append(transforms.RandomCrop(image_size))
            elif resize_strategy == "pad":
                # torchvision的Resize默认保持长宽比，但需要手动pad
                # 为了简化，这里使用Resize然后CenterCrop作为fallback
                transform_list.append(transforms.Resize(image_size))
                transform_list.append(transforms.CenterCrop(image_size))
            else:
                # 默认crop
                resize_scale = augmentation.get("resize_scale", 1.1)
                transform_list.append(transforms.Resize(int(image_size * resize_scale)))
                transform_list.append(transforms.RandomCrop(image_size))
            
            # 原有的random_crop选项（如果指定了且使用squash策略）
            if augmentation.get("random_crop", False) and resize_strategy == "squash":
                transform_list.append(transforms.RandomCrop(image_size))
            
            if augmentation.get("horizontal_flip", 0) > 0:
                transform_list.append(
                    transforms.RandomHorizontalFlip(p=augmentation["horizontal_flip"])
                )
            
            if augmentation.get("color_jitter", False):
                # 处理color_jitter可能是布尔值或字典的情况
                cj = augmentation.get("color_jitter_params", {})
                if not isinstance(cj, dict):
                    cj = {}
                transform_list.append(
                    transforms.ColorJitter(
                        brightness=cj.get("brightness", 0.2),
                        contrast=cj.get("contrast", 0.2),
                        saturation=cj.get("saturation", 0.2),
                        hue=cj.get("hue", 0.1),
                    )
                )
            
            transform_list.extend([
                transforms.ToTensor(),
            ])
            
            norm_mean = augmentation.get("normalize", {}).get("mean", [0.485, 0.456, 0.406])
            norm_std = augmentation.get("normalize", {}).get("std", [0.229, 0.224, 0.225])
            transform_list.append(transforms.Normalize(mean=norm_mean, std=norm_std))
            
            return transforms.Compose(transform_list)
    else:
        # Validation/test transforms (no augmentation)
        if ALBUMENTATIONS_AVAILABLE:
            norm_mean = [0.485, 0.456, 0.406]
            norm_std = [0.229, 0.224, 0.225]
            if augmentation and "normalize" in augmentation:
                norm_mean = augmentation["normalize"].get("mean", norm_mean)
                norm_std = augmentation["normalize"].get("std", norm_std)
            
            # 验证/测试时也使用保持长宽比的策略
            transform_list = []
            
            if resize_strategy == "squash":
                transform_list.append(A.Resize(image_size, image_size))
            elif resize_strategy == "crop":
                # 验证/测试时：短边resize到image_size，然后center crop
                transform_list.append(A.SmallestMaxSize(max_size=image_size))
                transform_list.append(A.CenterCrop(height=image_size, width=image_size))
            elif resize_strategy == "pad":
                transform_list.append(A.SmallestMaxSize(max_size=image_size))
                transform_list.append(
                    A.PadIfNeeded(
                        min_height=image_size,
                        min_width=image_size,
                        border_mode=0,
                        value=0,
                        mask_value=0
                    )
                )
            else:
                # 默认crop
                transform_list.append(A.SmallestMaxSize(max_size=image_size))
                transform_list.append(A.CenterCrop(height=image_size, width=image_size))
            
            transform_list.extend([
                A.Normalize(mean=norm_mean, std=norm_std),
                ToTensorV2(),
            ])
            
            return A.Compose(transform_list)
        else:
            norm_mean = [0.485, 0.456, 0.406]
            norm_std = [0.229, 0.224, 0.225]
            if augmentation and "normalize" in augmentation:
                norm_mean = augmentation["normalize"].get("mean", norm_mean)
                norm_std = augmentation["normalize"].get("std", norm_std)
            
            # 验证/测试时也使用保持长宽比的策略
            transform_list = []
            
            if resize_strategy == "squash":
                # 强制拉伸
                transform_list.append(transforms.Resize((image_size, image_size)))
            elif resize_strategy == "crop":
                # 保持长宽比，短边resize到image_size，然后center crop
                transform_list.append(transforms.Resize(image_size))
                transform_list.append(transforms.CenterCrop(image_size))
            elif resize_strategy == "pad":
                # 保持长宽比，短边resize到image_size，然后pad
                # torchvision没有直接的pad，需要自定义或使用Resize
                # 这里简化处理：先resize短边，然后pad
                transform_list.append(transforms.Resize(image_size))
                # 注意：torchvision的Resize默认保持长宽比，但需要手动pad
                # 为了简化，这里使用Resize然后CenterCrop作为fallback
                transform_list.append(transforms.CenterCrop(image_size))
            else:
                # 默认crop
                transform_list.append(transforms.Resize(image_size))
                transform_list.append(transforms.CenterCrop(image_size))
            
            transform_list.extend([
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std)
            ])
            
            return transforms.Compose(transform_list)


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
    
    # Get transforms
    # 如果augmentation配置了use_vit_transforms，使用ViT专用的transforms
    use_vit_transforms = augmentation and augmentation.get("use_vit_transforms", False)
    
    if use_vit_transforms:
        train_transform = get_vit_transforms("train", image_size, augmentation)
        val_transform = get_vit_transforms("val", image_size, augmentation)
        test_transform = get_vit_transforms("test", image_size, augmentation)
    else:
        # 默认使用Qwen VLM的transforms (None to keep PIL Images)
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
