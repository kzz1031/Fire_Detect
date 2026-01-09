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
![](pic/vit.png)
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

The model demonstrates reasonable performance on smoke detection (`start_fire`) but struggles with distinguishing actual flames from smoke, similar to the challenges observed in other experiments.

### 3. Swin Transformer Approach

#### 3.1 Model Architecture

We implemented a Swin Transformer (Shifted Window Transformer) for fire detection. Unlike ViT which uses global self-attention, Swin Transformer employs a hierarchical architecture with shifted window-based self-attention, making it more efficient and better suited for vision tasks.

**Key Components:**

1. **Patch Embedding**: Converts input images (224×224) into non-overlapping patches (4×4) and projects them into embedding space using a convolutional layer.

2. **Hierarchical Architecture**: The model consists of 4 stages, each with different resolutions:
   - Stage 1: 56×56 patches (after 4×4 patch embedding)
   - Stage 2: 28×28 patches (after first patch merging)
   - Stage 3: 14×14 patches (after second patch merging)
   - Stage 4: 7×7 patches (after third patch merging)

3. **Swin Transformer Blocks**: Each stage contains multiple Swin Transformer blocks with:
   - **Window-based Multi-head Self-Attention (W-MSA)**: Computes attention within fixed-size windows (7×7), reducing computational complexity from O(n²) to O(n) where n is the number of patches
   - **Shifted Window-based Multi-head Self-Attention (SW-MSA)**: Alternates with W-MSA, using shifted windows to enable cross-window connections while maintaining efficiency
   - **Relative Position Bias**: Adds learnable relative position encodings within windows, capturing spatial relationships more effectively than absolute position encodings
   - **MLP**: Two-layer feed-forward network with GELU activation
   - **Layer Normalization and Residual Connections**: Applied before attention and MLP layers

4. **Patch Merging**: Downsampling layers between stages that merge 2×2 neighboring patches, doubling the channel dimension while halving spatial resolution.

5. **Classification Head**: Global average pooling followed by a linear layer mapping to class logits.

**Model Variants:**
- `swin_tiny`: embed_dim=96, depths=[2,2,6,2], num_heads=[3,6,12,24]
- `swin_small`: embed_dim=96, depths=[2,2,18,2], num_heads=[3,6,12,24]
- `swin_base`: embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32]

**Advantages over ViT:**
- **Linear computational complexity**: Window-based attention scales linearly with image size, making it more efficient for high-resolution images
- **Hierarchical feature representation**: Multi-scale features capture both local and global patterns
- **Better inductive bias**: The shifted window mechanism and hierarchical structure provide stronger spatial locality priors

The implementation follows the principle of "good taste" - eliminating unnecessary complexity while maintaining the core architectural innovations that make Swin Transformer effective.

#### 3.2 Experimental Results

We conducted extensive experiments with different model sizes, hyperparameters, and training strategies:

**Model Size Comparison:**

| Configuration | Accuracy | Precision | Recall | F1 | No_fire F1 | Start_fire F1 | Fire F1 |
|---------------|----------|-----------|--------|----|-----------|---------------|---------|
| swin_tiny | 71.7% | 65.7% | 73.7% | 64.7% | 74.8% | 80.4% | 38.8% |
| swin_small | 73.0% | 66.8% | 77.5% | 67.2% | 74.9% | 80.7% | 46.2% |
| swin_base | 69.0% | 65.2% | 74.2% | 63.2% | 74.0% | 77.6% | 38.0% |

**Learning Rate Experiments:**

| Configuration | Accuracy | Precision | Recall | F1 | No_fire F1 | Start_fire F1 | Fire F1 |
|---------------|----------|-----------|--------|----|-----------|---------------|---------|
| swin_lr_1e-4 | 71.2% | 65.2% | 73.4% | 64.6% | 74.1% | 78.3% | 41.3% |
| swin_lr_2e-4 | 74.3% | 67.7% | 78.3% | 68.3% | 77.0% | 81.1% | 46.9% |
| swin_lr_5e-5 | 72.6% | 66.5% | 74.4% | 65.4% | 75.0% | 83.9% | 37.1% |

**Batch Size Experiments:**

| Configuration | Accuracy | Precision | Recall | F1 | No_fire F1 | Start_fire F1 | Fire F1 |
|---------------|----------|-----------|--------|----|-----------|---------------|---------|
| swin_bs32 | 73.9% | 67.8% | 76.9% | 67.0% | 76.0% | 84.6% | 40.6% |
| swin_bs64 | 72.6% | 67.6% | 76.3% | 65.8% | 74.3% | 84.3% | 38.9% |
| swin_bs128 | 73.0% | 66.1% | 73.3% | 65.5% | 75.7% | 80.2% | 40.7% |

**Regularization Experiments:**

| Configuration | Accuracy | Precision | Recall | F1 | No_fire F1 | Start_fire F1 | Fire F1 |
|---------------|----------|-----------|--------|----|-----------|---------------|---------|
| swin_dropout_0.0 | 71.7% | 65.8% | 73.6% | 64.6% | 75.7% | 79.8% | 38.2% |
| swin_dropout_0.1 | 72.6% | 65.8% | 73.1% | 64.9% | 74.8% | 82.4% | 37.5% |
| swin_dropout_0.2 | 71.2% | 66.5% | 75.0% | 64.7% | 74.3% | 80.9% | 38.9% |

**Loss Function Experiments:**

| Configuration | Accuracy | Precision | Recall | F1 | No_fire F1 | Start_fire F1 | Fire F1 |
|---------------|----------|-----------|--------|----|-----------|---------------|---------|
| swin_focal_loss | 72.6% | 66.9% | 76.0% | 65.9% | 74.8% | 83.0% | 40.0% |
| swin_class_weight | 71.7% | 66.7% | 75.6% | 65.2% | 73.6% | 81.3% | 40.6% |

**Key Observations:**
- Best performance achieved with learning rate 2e-4 (74.3% accuracy, 68.3% F1)
- `swin_small` shows better performance than `swin_tiny` and `swin_base`, suggesting optimal model capacity for this task
- Batch size 32 achieves the best balance (73.9% accuracy, 67.0% F1)
- `start_fire` category shows excellent performance across all configurations (F1: 77.6% - 84.6%), significantly better than ViT
- `fire` category remains challenging (F1: 37.1% - 46.9%), but `swin_lr_2e-4` achieves the best fire detection (46.9% F1)
- Dropout has minimal impact on performance, suggesting the model is not overfitting significantly
- Focal loss and class weighting show similar performance, with focal loss slightly better for `start_fire` detection

The hierarchical architecture and window-based attention mechanism of Swin Transformer prove particularly effective for detecting smoke (`start_fire`), capturing multi-scale features that help distinguish early-stage fire indicators.

### 4. Qwen2-VL-7B Pretrained Model Experiments

#### 4.1 Overview

We also experimented with the Qwen2-VL-7B pretrained vision-language model using zero-shot classification through prompt engineering. This approach leverages the model's pre-trained understanding of visual concepts without fine-tuning.

#### 4.2 Prompt Engineering Optimization Process

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

#### 4.3 Qwen2-VL Experimental Results Comparison

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

### 5. Final Ranking

No. 41

<img width="1950" height="2224" alt="image" src="https://github.com/user-attachments/assets/b756a864-723d-4cb0-bc4a-29a01d978f8d" />

### 6. Training Logs (wandb)
<img width="5040" height="2532" alt="image" src="https://github.com/user-attachments/assets/341e6193-fc7d-4b82-b1e5-9919058544aa" />
