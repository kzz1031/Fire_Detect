# Training Scripts

## Qwen VLM Training

Train the Qwen VLM model for fire detection:

```bash
cd scripts/train
python train_qwen_vlm.py
```

### Configuration

The training uses Hydra for configuration management. All configs are in `configs/qwen_vlm/`.

### WandB Setup

Before training, make sure to:
1. Login to WandB: `wandb login`
2. Update `configs/qwen_vlm/config.yaml` with your WandB entity/username

### Custom Configuration

Override config values via command line:

```bash
python train_qwen_vlm.py \
    training.epochs=20 \
    training.learning_rate=1e-4 \
    data.batch_size=8 \
    wandb.name="my-experiment"
```

### Outputs

- Checkpoints: `checkpoints/qwen_vlm/`
- Logs: `logs/qwen_vlm/`
- Results: `results/qwen_vlm/`
- WandB: Project `fire-detection-qwen`

