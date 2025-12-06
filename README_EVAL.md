# 预训练模型评估指南

## 评估预训练 Qwen2-VL-7B 在测试集上的表现

这个脚本使用**未微调**的 Qwen2-VL-7B 模型在测试集上进行评估，用于了解预训练模型的 baseline 性能。

### 快速开始

```bash
# 方法1: 使用脚本
cd scripts/eval
bash run_pretrained_eval.sh

# 方法2: 直接运行 Python
python scripts/eval/eval_qwen2_vl_pretrained.py \
    --model_name "Qwen/Qwen2-VL-7B-Instruct" \
    --test_dir "data/test" \
    --output_dir "results/qwen_vlm/pretrained" \
    --device "cuda"
```

### 参数说明

- `--model_name`: 模型名称（默认: "Qwen/Qwen2-VL-7B-Instruct"）
- `--test_dir`: 测试集目录（默认: "data/test"）
- `--output_dir`: 结果输出目录（默认: "results/qwen_vlm/pretrained"）
- `--device`: 设备（默认: "cuda"）
- `--max_new_tokens`: 生成的最大 token 数（默认: 10）

### 输出结果

评估完成后，会在 `output_dir` 生成以下文件：

1. **pretrained_metrics.json**: 评估指标
   - accuracy, precision, recall, f1
   - 每个类别的详细指标

2. **pretrained_results.json**: 每个样本的详细结果
   - 图像路径
   - 真实标签
   - 预测标签
   - 是否正确

3. **pretrained_confusion_matrix.txt**: 混淆矩阵

### 工作原理

1. **加载模型**: 从 HuggingFace 加载预训练的 Qwen2-VL-7B 模型
2. **加载测试集**: 从 `data/test/` 目录加载所有测试图像
3. **分类提示**: 对每张图像使用以下提示：
   ```
   "Look at this image carefully. 
    Classify it into one of these three categories: 
    'fire', 'start_fire', or 'no_fire'. 
    Respond with only one word: fire, start_fire, or no_fire."
   ```
4. **生成预测**: 模型生成文本响应，解析为类别
5. **计算指标**: 计算准确率、精确率、召回率、F1 分数等

### 注意事项

- **首次运行**: 会从 HuggingFace 下载模型（约 14GB），需要网络连接
- **显存需求**: Qwen2-VL-7B 需要约 16GB GPU 显存
- **运行时间**: 取决于测试集大小，每张图像约需 1-3 秒
- **结果解析**: 模型可能返回各种格式的文本，脚本会尝试智能解析

### 示例输出

```
Evaluation Results:
==================================================
Accuracy: 0.7234
Precision: 0.7123
Recall: 0.7234
F1 Score: 0.7178

Per-class metrics:
  fire: Precision=0.8500, Recall=0.7500, F1=0.7969
  start_fire: Precision=0.6500, Recall=0.7200, F1=0.6833
  no_fire: Precision=0.7800, Recall=0.7000, F1=0.7377
```

### 与微调模型对比

运行此评估后，可以：
1. 了解预训练模型的 baseline 性能
2. 与微调后的模型性能进行对比
3. 分析哪些类别容易混淆
4. 为微调策略提供参考

