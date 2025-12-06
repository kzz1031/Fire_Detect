# Fire Detection Challenge

Image classification challenge: fire, start of fire, and no fire detection.

## Project Structure

```
Fire_Detect/
├── data/                    # Dataset
│   ├── FIRE_DATABASE_1/     # Database 1
│   ├── FIRE_DATABASE_2/     # Database 2
│   ├── FIRE_DATABASE_3/     # Database 3
│   ├── test/                # Test set
│   ├── processed/           # Processed data (ignored by git)
│   └── splits/              # Train/val/test splits (ignored by git)
│
├── models/                  # Model implementations
│   ├── vit/                 # Vision Transformer model
│   └── qwen_vlm/            # Qwen VLM fine-tuning
│
├── configs/                 # Configuration files
│   ├── vit/                 # ViT configs
│   └── qwen_vlm/            # Qwen VLM configs
│
├── utils/                   # Utility functions
│   ├── data_loader.py       # Data loading utilities
│   ├── transforms.py        # Data augmentation
│   ├── metrics.py           # Evaluation metrics
│   └── visualization.py     # Visualization tools
│
├── scripts/                 # Training and evaluation scripts
│   ├── train/               # Training scripts
│   │   ├── train_vit.py
│   │   └── train_qwen_vlm.py
│   ├── eval/                # Evaluation scripts
│   │   ├── eval_vit.py
│   │   └── eval_qwen_vlm.py
│   └── inference/           # Inference scripts
│
├── checkpoints/             # Model checkpoints (ignored by git)
│   ├── vit/
│   └── qwen_vlm/
│
├── logs/                    # Training logs (ignored by git)
│   ├── vit/
│   └── qwen_vlm/
│
└── results/                 # Results and predictions (ignored by git)
    ├── vit/
    └── qwen_vlm/
```

## Models

### 1. Vision Transformer (ViT)
- Base model: ViT-B/16 pretrained on ImageNet
- Fine-tuning approach for fire detection

### 2. Qwen VLM
- Base model: Qwen-VL
- Vision-Language model fine-tuning for fire detection

## Dataset

Three categories:
- **Fire**: Images containing fire
- **Beginning of Fire**: Images showing the start of fire
- **No Fire**: Images with no trace of fire

## Evaluation Metric

Accuracy: Percentage of images correctly classified.

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare data:
```bash
python scripts/prepare_data.py
```

3. Train ViT model:
```bash
python scripts/train/train_vit.py --config configs/vit/config.yaml
```

4. Train Qwen VLM:
```bash
python scripts/train/train_qwen_vlm.py --config configs/qwen_vlm/config.yaml
```

5. Evaluate models:
```bash
python scripts/eval/eval_vit.py --checkpoint checkpoints/vit/best.pth
python scripts/eval/eval_qwen_vlm.py --checkpoint checkpoints/qwen_vlm/best.pth
```

## Citation

Sidi Ahmed Mahmoudi & Aurélie Cools. HANDS ON AI @UMONS 2025-2026. https://kaggle.com/competitions/hands-on-ai-umons-2025-2026, 2025. Kaggle.
