# Quick Start Guide - Qwen VLM Training

## 安装依赖

```bash
pip install -r requirements.txt
```

## WandB 设置

1. 登录 WandB：
```bash
wandb login
```

2. 在 `configs/qwen_vlm/config.yaml` 中设置你的 WandB entity（如果需要）

## 训练模型

### 基本训练

```bash
cd scripts/train
python train_qwen_vlm.py
```

### 自定义配置

通过命令行覆盖配置：

```bash
python train_qwen_vlm.py \
    training.epochs=20 \
    training.learning_rate=1e-4 \
    data.batch_size=8 \
    wandb.name="my-experiment"
```

### 使用不同的模型

```bash
# 使用 Qwen2-VL-7B (新版本 7B)
python train_qwen_vlm.py \
    model.model_name="Qwen/Qwen2-VL-7B-Instruct" \
    model.use_peft=true

# 使用 Qwen2-VL-2B (更小更快，显存需求低)
python train_qwen_vlm.py \
    model.model_name="Qwen/Qwen2-VL-2B-Instruct" \
    model.use_peft=true

# 使用原始 Qwen-VL (基于 Qwen-7B)
python train_qwen_vlm.py \
    model.model_name="Qwen/Qwen-VL" \
    model.use_peft=true
```

### 模型大小对比

- **Qwen/Qwen-VL**: ~7B 参数（原始版本，基于 Qwen-7B）
- **Qwen/Qwen-VL-Chat**: ~7B 参数（对话优化版本）
- **Qwen/Qwen2-VL-2B-Instruct**: ~2B 参数（更小，显存需求低）
- **Qwen/Qwen2-VL-7B-Instruct**: ~7B 参数（新版本，性能更好）
- **Qwen/Qwen2-VL-72B-Instruct**: ~72B 参数（最大版本，需要大量显存）

## 配置说明

### 模型配置 (`configs/qwen_vlm/model/qwen_vlm.yaml`)

- `model_name`: Qwen VLM 模型名称
- `num_classes`: 分类数量（3：fire, start_fire, no_fire）
- `use_peft`: 是否使用 LoRA 微调（推荐开启以节省显存）
- `lora_r`: LoRA 的 rank
- `lora_alpha`: LoRA 的 alpha 参数

### 数据配置 (`configs/qwen_vlm/data/fire_detection.yaml`)

- `root_dir`: 数据根目录
- `databases`: 使用的数据库列表
- `classes`: 类别列表
- `batch_size`: 批次大小（根据显存调整）
- `augmentation`: 数据增强配置

### 训练配置 (`configs/qwen_vlm/training/qwen_training.yaml`)

- `epochs`: 训练轮数
- `learning_rate`: 学习率
- `gradient_accumulation_steps`: 梯度累积步数（用于模拟更大的 batch size）
- `mixed_precision`: 是否使用混合精度训练（节省显存）
- `early_stopping`: 早停配置

## 输出文件

训练完成后，会在以下目录生成文件：

- `checkpoints/qwen_vlm/`: 模型检查点
  - `best_model.pth`: 最佳模型
  - `checkpoint_epoch_*.pth`: 定期保存的检查点
- `logs/qwen_vlm/`: 训练日志
- `results/qwen_vlm/`: 测试结果
  - `test_results.json`: 测试集指标
- WandB: 在线训练监控和可视化

## 常见问题

### 显存不足

1. 减小 `batch_size`
2. 增加 `gradient_accumulation_steps`
3. 启用 `mixed_precision: true`
4. 启用 `use_peft: true`（LoRA 微调）

### 模型加载失败

确保：
1. 已安装 `transformers` 和 `peft`
2. 有足够的网络连接下载模型（或使用本地模型路径）
3. 模型名称正确

### 数据加载错误

检查：
1. 数据路径是否正确
2. 数据库文件夹是否存在
3. 类别文件夹名称是否匹配

## 下一步

训练完成后，可以使用 `scripts/eval/eval_qwen_vlm.py` 评估模型性能。

