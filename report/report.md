# Kaggle: Fire Detection
22300240022 王镜凯, 23302010087 鲁蓉希, 张湄悦

## Experiment Report: Fire Detection

### 1. Project Overview

This project aims to classify images into three fire-related categories:
- `no_fire`: No fire or smoke visible
- `start_fire`: Smoke only visible (early-stage fire, no visible flames)
- `fire`: Visible flames present

### 2. Vision Transformer (ViT) Approach

#### 2.1 Model Architecture

We implemented a Vision Transformer from scratch for fire detection. The model follows the standard ViT architecture:

**Key Components:**
- **Patch Embedding**: Divides input images (224×224) into 16×16 patches and projects them into embedding space
- **CLS Token**: A learnable classification token prepended to the patch sequence
- **Position Embedding**: Learnable positional encodings for spatial information
- **Transformer Blocks**: Stack of 12 transformer layers, each containing:
  - Multi-head self-attention mechanism
  - Feed-forward MLP with GELU activation
  - Layer normalization and residual connections
- **Classification Head**: Linear layer that maps the CLS token to class logits

**Model Variants:**
- `vit_tiny`: embed_dim=192, depth=12, num_heads=3
- `vit_small`: embed_dim=384, depth=12, num_heads=6
- `vit_base`: embed_dim=768, depth=12, num_heads=12

The implementation emphasizes simplicity and eliminates unnecessary complexity, following the principle of "good taste" in code design.

#### 2.2 Experimental Results

We conducted several experiments with different hyperparameters:

| Configuration | Accuracy | Precision | Recall | F1 | No_fire F1 | Start_fire F1 | Fire F1 |
|---------------|----------|-----------|--------|----|-----------|---------------|---------|
| vit (default) | 70.4% | 81.7% | 70.4% | 72.4% | 73.9% | 77.8% | 37.5% |
| vit_1e-4 (lr=1e-4) | 73.9% | 82.4% | 73.9% | 74.8% | 76.7% | 77.3% | 50.0% |
| vit_5e-5 (lr=5e-5) | 69.9% | 77.4% | 69.9% | 71.4% | 72.9% | 74.3% | 47.6% |
| vit_dropout_enhanced | 73.0% | 80.7% | 73.0% | 74.9% | 78.3% | 76.6% | 43.3% |
| vit_bs_64 (batch_size=64) | 73.0% | 78.1% | 73.0% | 74.2% | 77.1% | 75.4% | 48.1% |

**Key Observations:**
- Best performance achieved with learning rate 1e-4 (73.9% accuracy)
- Increasing batch size to 64 shows similar overall accuracy (73.0%) but slightly lower `fire` precision (36.1% vs 38.2% in lr=1e-4)
- `start_fire` category shows good recall (87.9% in default config, 91.5% in lr=1e-4, 80.5% in bs_64)
- `fire` category remains challenging with lower precision (26.1% - 38.2%)
- `no_fire` category has high precision (88.6% - 98.8%) but moderate recall (59.5% - 68.3%)

The model demonstrates reasonable performance on smoke detection (`start_fire`) but struggles with distinguishing actual flames from smoke, similar to the challenges observed in the Qwen2-VL experiments.

### 3. Qwen2-VL-7B Pretrained Model Experiments

#### 3.1 Overview

We also experimented with the Qwen2-VL-7B pretrained vision-language model using zero-shot classification through prompt engineering. This approach leverages the model's pre-trained understanding of visual concepts without fine-tuning.

#### 3.2 Prompt Engineering Optimization Process

**Initial Version (before_PE)**
Simple prompt used:
```
"Look at this image carefully. Classify it into one of these three categories: 'fire', 'start_fire', or 'no_fire'. Respond with only one word: fire, start_fire, or no_fire."
```

**Problem Analysis:**
- Model tended to classify all fire-related images as `fire`, ignoring the `start_fire` category
- `start_fire` recall was only 1.2%, basically unable to identify images with smoke only

**First Optimization (after_PE)**
Improved prompt with detailed category descriptions:
```
"Look at this image carefully and analyze it for fire-related content. Classify it into exactly ONE of these three categories:
- 'no_fire': No fire, smoke, or fire-related activity visible in the image
- 'start_fire': Smoke visible but no visible flames or active fire. This indicates early-stage fire or smoldering
- 'fire': Visible flames, active burning fire, or significant fire activity

Focus on distinguishing between smoke-only (start_fire) and actual flames (fire). Respond with exactly one word: no_fire, start_fire, or fire."
```

**Optimization Results:**
- `start_fire` recall improved from 1.2% to 47.6%
- Overall accuracy improved from 62.8% to 80.1%

**Second Optimization (after_PEv2)**
Further strengthened smoke vs. flame distinction prompts:
```
"You are a fire detection expert. Analyze this image carefully for signs of fire.

CRITICAL DISTINCTION - Choose exactly ONE category:
- 'no_fire': NO smoke, NO flames, NO fire activity of any kind
- 'start_fire': SMOKE ONLY - visible smoke but ABSOLUTELY NO visible flames, sparks, or active burning
- 'fire': FLAMES VISIBLE - you can see actual flames, fire, or active burning

If you see smoke without flames, it MUST be 'start_fire'.
Only classify as 'fire' if you can clearly see flames or active fire.

Respond with exactly one word: no_fire, start_fire, or fire."
```

Simultaneously optimized post-processing logic to prevent `start_fire` from being misclassified as `fire`.

**Optimization Results:**
- `start_fire` recall further improved to 63.4%
- Overall accuracy improved to 85.8%
- `fire` category precision improved from 30.2% to 41.0%

#### 3.3 Qwen2-VL Experimental Results Comparison

| Version | Accuracy | Fire Precision | Fire Recall | Start_fire Recall | No_fire Recall |
|---------|----------|----------------|-------------|-------------------|---------------|
| before_PE | 62.8% | 19.1% | 94.4% | 1.2% | 98.4% |
| after_PE | 80.1% | 30.2% | 88.9% | 47.6% | 100% |
| after_PEv2 | 85.8% | 41.0% | 88.9% | 63.4% | 100% |

**Key Findings:**
- Prompt engineering significantly improved the model's ability to distinguish between smoke and flames
- The critical distinction between `start_fire` (smoke only) and `fire` (visible flames) required explicit emphasis in the prompt
- Post-processing logic was essential to prevent misclassification
- Despite improvements, `fire` category precision remains relatively low (41.0%), indicating confusion between smoke and flames
