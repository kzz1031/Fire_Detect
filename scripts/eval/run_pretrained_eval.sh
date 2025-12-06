#!/bin/bash
# Quick script to evaluate pretrained Qwen2-VL-7B on test set

cd "$(dirname "$0")/../.."

python scripts/eval/eval_qwen2_vl_pretrained.py \
    --model_name "Qwen/Qwen2-VL-7B-Instruct" \
    --test_dir "data/test" \
    --output_dir "results/qwen_vlm/pretrained" \
    --device "cuda" \
    --max_new_tokens 10

