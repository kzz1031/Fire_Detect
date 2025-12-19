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
│   ├── swin/                # Swin Transformer model
│   └── qwen_vlm/            # Qwen VLM fine-tuning
│
├── configs/                 # Configuration files
│   ├── vit/                 # ViT configs
│   ├── swin/                # Swin configs
│   └── qwen_vlm/            # Qwen VLM configs
│
├── utils/                   # Utility functions
│   ├── data_loader.py       # Data loading utilities
│   ├── transforms.py        # Data augmentation
│   ├── metrics.py           # Evaluation metrics
│   └── visualization.py    # Visualization tools
│
├── scripts/                 # Training and evaluation scripts
│   ├── train/               # Training scripts
│   │   ├── train_vit.py
│   │   ├── train_swin.py
│   │   └── train_qwen_vlm.py
│   ├── eval/                # Evaluation scripts
│   │   ├── eval_vit.py
│   │   └── eval_qwen_vlm.py
│   ├── inference/           # Inference scripts
│   │   ├── infer_vit.py
│   │   └── infer_swin.py
│   └── utils/               # Utility scripts
│       └── upload_to_hf.py  # Upload checkpoints to Hugging Face
│
├── checkpoints/             # Model checkpoints (ignored by git)
│   ├── vit/
│   ├── swin/
│   └── qwen_vlm/
│
├── logs/                    # Training logs (ignored by git)
│   ├── vit/
│   ├── swin/
│   └── qwen_vlm/
│
└── results/                 # Results and predictions (ignored by git)
    ├── vit/
    ├── swin/
    └── qwen_vlm/
```

## Models

### 1. Vision Transformer (ViT)
- **Architecture**: Custom ViT implementation from scratch
- **Variants**: `vit_tiny`, `vit_small`, `vit_base`
- **Features**: 
  - Patch-based image processing
  - Multi-head self-attention
  - Learnable positional embeddings
  - Dropout and drop path regularization

### 2. Swin Transformer
- **Architecture**: Hierarchical Swin Transformer with shifted windows
- **Variants**: `swin_tiny`, `swin_small`, `swin_base`
- **Features**:
  - Window-based attention (linear complexity)
  - Hierarchical feature representation
  - Shifted window mechanism for cross-window connections

### 3. Qwen VLM
- **Base model**: Qwen2-VL-7B
- **Approach**: Zero-shot classification with prompt engineering

## Dataset

Three categories:
- **fire**: Images containing visible flames
- **start_fire**: Images showing smoke only (early-stage fire, no visible flames)
- **no_fire**: Images with no trace of fire or smoke

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/kzz1031/Fire_Detect.git
cd Fire_Detect
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare data
> Datas are from Kaggle (https://kaggle.com/competitions/hands-on-ai-umons-2025-2026)
Ensure your data is organized as follows:
```
data/
├── FIRE_DATABASE_1/
│   ├── fire/
│   ├── start_fire/
│   └── no_fire/
├── FIRE_DATABASE_2/
│   ├── fire/
│   ├── start_fire/
│   └── no_fire/
├── FIRE_DATABASE_3/
│   ├── fire/
│   ├── start_fire/
│   └── no_fire/
└── test/
    ├── fire/
    ├── start_fire/
    └── no_fire/
```

## Training

### Training ViT Model

#### 1. Configure training parameters

Edit `configs/vit/config.yaml` to customize:
- Model architecture (`vit_tiny`, `vit_small`, `vit_base`)
- Image size (default: 224)
- Batch size (default: 64)
- Learning rate (default: 5e-5)
- Number of epochs (default: 50)
- Data augmentation settings
- Optimizer and scheduler settings

Example configuration:
```yaml
model:
  name: "vit_base"
  num_classes: 3
  image_size: 224
  dropout: 0.0
  drop_path: 0.0

data:
  root_dir: "data"
  batch_size: 64
  num_workers: 4
  augmentation:
    enabled: true
    resize_strategy: "crop"  # Keep aspect ratio

training:
  epochs: 50
  learning_rate: 5e-5
  optimizer: "adamw"
  scheduler: "cosine"
  warmup_epochs: 10
```

#### 2. Start training

```bash
python scripts/train/train_vit.py
```

The script will:
- Load data from `data/` directory
- Split data into train/val/test sets (default: 80%/20%/0%)
- Apply data augmentation during training
- Save checkpoints to `checkpoints/vit/`
- Save logs to `logs/vit/`
- Log metrics to TensorBoard and Wandb (if enabled)

#### 3. Monitor training

**TensorBoard:**
```bash
tensorboard --logdir logs/vit
```

**Wandb:**
- Enable in `configs/vit/config.yaml`:
```yaml
wandb:
  enabled: true
  project: "fire_detection_vit"
```
- View at: https://wandb.ai

#### 4. Training outputs

- **Checkpoints**: Saved in `checkpoints/vit/`
  - `best.pth`: Best model based on validation accuracy
  - `latest.pth`: Latest checkpoint
  - `config_*.yaml`: Training configuration snapshots

- **Logs**: Saved in `logs/vit/`
  - TensorBoard event files
  - Wandb logs (if enabled)

### Training Swin Transformer

#### 1. Configure training parameters

Edit `configs/swin/config.yaml`:

```yaml
model:
  name: "swin_small"  # swin_tiny, swin_small, swin_base
  num_classes: 3
  image_size: 224

data:
  root_dir: "data"
  batch_size: 32
  # ... similar to ViT config
```

#### 2. Start training

```bash
python scripts/train/train_swin.py
```

Training process is similar to ViT.

## Downloading Checkpoints

### Option 1: Download from Hugging Face (Recommended)

Checkpoints are available on Hugging Face: [kzzwang/fire_detect](https://huggingface.co/kzzwang/fire_detect)

#### Download ViT checkpoint:

```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Download ViT checkpoint
python -c "
from huggingface_hub import hf_hub_download
import torch
from pathlib import Path

# Download checkpoint
checkpoint_path = hf_hub_download(
    repo_id='kzzwang/fire_detect',
    filename='vit/best.pth',
    local_dir='checkpoints',
    local_dir_use_symlinks=False
)
print(f'Downloaded to: {checkpoint_path}')
"
```

Or use the command line:
```bash
huggingface-cli download kzzwang/fire_detect vit/best.pth --local-dir checkpoints/vit
```

#### Download Swin checkpoint:

```bash
huggingface-cli download kzzwang/fire_detect swin/best.pth --local-dir checkpoints/swin
```

#### Download all checkpoints:

```bash
# Download entire repository
huggingface-cli download kzzwang/fire_detect --local-dir checkpoints
```

### Option 2: Manual download

1. Visit: https://huggingface.co/kzzwang/fire_detect
2. Navigate to the model folder (`vit/` or `swin/`)
3. Download `best.pth`
4. Place it in `checkpoints/vit/` or `checkpoints/swin/`

### Option 3: Use local checkpoints

If you've trained locally, checkpoints are already in:
- `checkpoints/vit/best.pth`
- `checkpoints/swin/best.pth`

## Inference

### Inference with ViT

#### 1. Basic inference on test set

```bash
python scripts/inference/infer_vit.py
```

This will:
- Load the best checkpoint from `checkpoints/vit/best.pth`
- Run inference on the test set from `data/test/`
- Save results to `results/vit/`

#### 2. Custom checkpoint and output directory

```bash
python scripts/inference/infer_vit.py \
    --checkpoint checkpoints/vit/best.pth \
    --output-dir results/vit_test
```

#### 3. Custom config file

```bash
python scripts/inference/infer_vit.py \
    --config configs/vit/config.yaml \
    --checkpoint checkpoints/vit/best.pth
```

#### 4. Inference outputs

Results are saved in the output directory:
- `best_results.json`: Detailed predictions for each image
  ```json
  {
    "image_path": "data/test/fire/image001.jpg",
    "true_label": "fire",
    "predicted_label": "fire",
    "correct": true,
    "probabilities": {
      "no_fire": 0.05,
      "start_fire": 0.15,
      "fire": 0.80
    }
  }
  ```
- `best_metrics.json`: Overall metrics
  ```json
  {
    "accuracy": 0.739,
    "precision": 0.824,
    "recall": 0.739,
    "f1": 0.748,
    "no_fire_precision": 0.886,
    "no_fire_recall": 0.683,
    "no_fire_f1": 0.772,
    ...
  }
  ```
- `best_confusion_matrix.txt`: Confusion matrix

### Inference with Swin Transformer

```bash
python scripts/inference/infer_swin.py \
    --checkpoint checkpoints/swin/best.pth \
    --output-dir results/swin_test
```

### Inference on single image

You can modify the inference script to process a single image:

```python
from pathlib import Path
import torch
from PIL import Image
from models.vit import build_vit
from utils.data_loader import get_vit_transforms

# Load model
model = build_vit("vit_base", num_classes=3, img_size=224)
checkpoint = torch.load("checkpoints/vit/best.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load and preprocess image
transform = get_vit_transforms("test", 224, None)
image = Image.open("path/to/image.jpg").convert('RGB')
image_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(image_tensor)
    probs = torch.softmax(output, dim=1)
    pred = torch.argmax(output, dim=1).item()

classes = ['no_fire', 'start_fire', 'fire']
print(f"Prediction: {classes[pred]}")
print(f"Probabilities: {dict(zip(classes, probs[0].tolist()))}")
```

## Evaluation

### Evaluate on test set

```bash
# ViT
python scripts/eval/eval_vit.py --checkpoint checkpoints/vit/best.pth

# Swin
python scripts/eval/eval_swin.py --checkpoint checkpoints/swin/best.pth
```

## Configuration Guide

### Key Configuration Parameters

#### Model Configuration
- `model.name`: Model variant (`vit_base`, `swin_small`, etc.)
- `model.image_size`: Input image size (224, 384, 512)
- `model.dropout`: Dropout rate (0.0-0.5)
- `model.drop_path`: Drop path rate for regularization

#### Data Configuration
- `data.batch_size`: Batch size for training
- `data.num_workers`: Number of data loading workers
- `data.augmentation.enabled`: Enable/disable data augmentation
- `data.augmentation.resize_strategy`: 
  - `"crop"`: Keep aspect ratio, then crop (recommended)
  - `"pad"`: Keep aspect ratio, then pad
  - `"squash"`: Force resize (not recommended)

#### Training Configuration
- `training.epochs`: Number of training epochs
- `training.learning_rate`: Initial learning rate
- `training.optimizer`: Optimizer (`adamw`, `adam`, `sgd`)
- `training.scheduler`: Learning rate scheduler (`cosine`, `step`)
- `training.warmup_epochs`: Warmup epochs for learning rate
- `training.use_focal_loss`: Use Focal Loss for class imbalance
- `training.use_class_weights`: Use weighted loss for class imbalance
- `training.label_smoothing`: Label smoothing coefficient

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce `batch_size` in config
   - Reduce `image_size`
   - Use smaller model variant

2. **Data loading errors**
   - Check data directory structure
   - Verify image file formats
   - Check `num_workers` (set to 0 if issues)

3. **Checkpoint not found**
   - Verify checkpoint path
   - Download from Hugging Face if needed
   - Check checkpoint directory structure

4. **Import errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python path includes project root

## Citation

Sidi Ahmed Mahmoudi & Aurélie Cools. HANDS ON AI @UMONS 2025-2026. https://kaggle.com/competitions/hands-on-ai-umons-2025-2026, 2025. Kaggle.
