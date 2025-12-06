"""
Training script for Qwen VLM model with Hydra configuration and WandB logging
"""
import os
import sys
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, get_linear_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup as get_scheduler
import wandb
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.qwen_vlm.model import QwenVLMForFireDetection
from utils.data_loader import create_data_loaders
from utils.metrics import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs/qwen_vlm", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    
    # Set random seeds for reproducibility
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    # Setup device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directories
    output_dir = Path(cfg.paths.output_dir)
    checkpoint_dir = Path(cfg.paths.checkpoint_dir)
    log_dir = Path(cfg.paths.log_dir)
    result_dir = Path(cfg.paths.result_dir)
    
    for dir_path in [output_dir, checkpoint_dir, log_dir, result_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, f)
    logger.info(f"Saved config to {config_path}")
    
    # Initialize WandB
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.get("entity"),
        name=cfg.wandb.name,
        tags=cfg.wandb.get("tags", []),
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=str(log_dir),
    )
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        root_dir=cfg.data.root_dir,
        databases=cfg.data.databases,
        classes=cfg.data.classes,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        image_size=cfg.model.image_size,
        augmentation=cfg.data.get("augmentation"),
        seed=cfg.seed,
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = QwenVLMForFireDetection(
        model_name=cfg.model.model_name,
        num_classes=cfg.model.num_classes,
        max_length=cfg.model.max_length,
        image_size=cfg.model.image_size,
        freeze_vision_encoder=cfg.model.get("freeze_vision_encoder", False),
        freeze_language_model=cfg.model.get("freeze_language_model", False),
        hidden_size=cfg.model.get("hidden_size", 1024),
        dropout=cfg.model.get("dropout", 0.1),
        use_peft=cfg.model.get("use_peft", True),
        peft_type=cfg.model.get("peft_type", "lora"),
        lora_r=cfg.model.get("lora_r", 16),
        lora_alpha=cfg.model.get("lora_alpha", 32),
        lora_dropout=cfg.model.get("lora_dropout", 0.1),
        target_modules=cfg.model.get("target_modules", None),
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    wandb.config.update({
        "total_params": total_params,
        "trainable_params": trainable_params,
    })
    
    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    
    # Calculate total training steps
    num_training_steps = len(train_loader) * cfg.training.epochs
    if cfg.training.gradient_accumulation_steps > 1:
        num_training_steps //= cfg.training.gradient_accumulation_steps
    
    # Setup learning rate scheduler
    warmup_steps = cfg.training.get("warmup_steps", 0)
    if warmup_steps == 0 and cfg.training.get("warmup_ratio", 0) > 0:
        warmup_steps = int(num_training_steps * cfg.training.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # Mixed precision training
    use_amp = cfg.training.get("mixed_precision", False) or cfg.training.get("fp16", False)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Training loop
    best_val_accuracy = 0.0
    global_step = 0
    patience_counter = 0
    
    logger.info("Starting training...")
    for epoch in range(cfg.training.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{cfg.training.epochs}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            images, labels, _ = batch
            
            # Images should already be PIL Images from data loader
            if not isinstance(images, list):
                # If they're tensors, convert to PIL
                from torchvision.transforms import ToPILImage
                to_pil = ToPILImage()
                images_list = [to_pil(img) for img in images]
            else:
                images_list = images
            
            labels = labels.to(device)
            
            # Forward pass with mixed precision
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images_list, labels=labels)
                    loss = outputs["loss"]
                    loss = loss / cfg.training.gradient_accumulation_steps
            else:
                outputs = model(images_list, labels=labels)
                loss = outputs["loss"]
                loss = loss / cfg.training.gradient_accumulation_steps
            
            # Backward pass
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % cfg.training.gradient_accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        cfg.training.get("max_grad_norm", 1.0)
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        cfg.training.get("max_grad_norm", 1.0)
                    )
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            # Accumulate metrics
            train_loss += loss.item() * cfg.training.gradient_accumulation_steps
            logits = outputs["logits"]
            predictions = torch.argmax(logits, dim=-1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            # Update progress bar
            current_loss = train_loss / (batch_idx + 1)
            current_acc = train_correct / train_total
            progress_bar.set_postfix({
                "loss": f"{current_loss:.4f}",
                "acc": f"{current_acc:.4f}"
            })
            
            # Log to WandB
            if global_step % cfg.training.get("logging_steps", 50) == 0:
                wandb.log({
                    "train/loss": current_loss,
                    "train/accuracy": current_acc,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "global_step": global_step,
                })
        
        # Calculate epoch metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = train_correct / train_total
        
        logger.info(
            f"Epoch {epoch+1}/{cfg.training.epochs} - "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}"
        )
        
        # Validation phase
        if (epoch + 1) % cfg.training.get("eval_steps", 500) == 0 or (epoch + 1) == cfg.training.epochs:
            logger.info("Running validation...")
            val_metrics, _, _, _ = evaluate_model(
                model,
                val_loader,
                device,
                class_names=cfg.data.classes
            )
            
            val_accuracy = val_metrics["accuracy"]
            logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
            
            # Log to WandB
            wandb.log({
                "val/accuracy": val_accuracy,
                "val/precision": val_metrics["precision"],
                "val/recall": val_metrics["recall"],
                "val/f1": val_metrics["f1"],
                "epoch": epoch + 1,
            })
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                
                checkpoint_path = checkpoint_dir / "best_model.pth"
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_accuracy": best_val_accuracy,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                }, checkpoint_path)
                logger.info(f"Saved best model to {checkpoint_path}")
                
                wandb.run.summary["best_val_accuracy"] = best_val_accuracy
            else:
                patience_counter += 1
            
            # Early stopping
            if cfg.training.early_stopping.get("enabled", False):
                if patience_counter >= cfg.training.early_stopping.get("patience", 3):
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Save checkpoint periodically
        if (epoch + 1) % cfg.training.get("save_steps", 1000) == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_accuracy": val_metrics.get("accuracy", 0.0),
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Final evaluation on test set
    logger.info("Running final evaluation on test set...")
    test_metrics, test_predictions, test_labels, test_confusion = evaluate_model(
        model,
        test_loader,
        device,
        class_names=cfg.data.classes
    )
    
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
    logger.info(f"Test F1: {test_metrics['f1']:.4f}")
    
    # Log final metrics to WandB
    wandb.log({
        "test/accuracy": test_metrics["accuracy"],
        "test/precision": test_metrics["precision"],
        "test/recall": test_metrics["recall"],
        "test/f1": test_metrics["f1"],
    })
    
    wandb.run.summary.update({
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
    })
    
    # Save final results
    import json
    results_path = result_dir / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Saved test results to {results_path}")
    
    wandb.finish()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
