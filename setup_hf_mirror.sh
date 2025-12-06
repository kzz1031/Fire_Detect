#!/bin/bash
# Setup HuggingFace mirror for China users

# Option 1: Use hf-mirror.com (recommended for China)
export HF_ENDPOINT="https://hf-mirror.com"

# Option 2: Use other mirrors (uncomment if needed)
# export HF_ENDPOINT="https://huggingface.co"  # Original
# export HF_ENDPOINT="https://hf.co"  # Alternative

# Also set for huggingface_hub
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "HuggingFace endpoint set to: $HF_ENDPOINT"
echo "Run: source setup_hf_mirror.sh before running scripts"

