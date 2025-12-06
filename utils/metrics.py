"""
Evaluation metrics for fire detection
"""
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: list = None
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        predictions: Predicted class indices
        labels: True class indices
        class_names: Optional list of class names
    
    Returns:
        Dictionary of metrics
    """
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    
    # Per-class metrics
    if class_names:
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        for idx, class_name in enumerate(class_names):
            metrics[f"{class_name}_precision"] = float(precision_per_class[idx])
            metrics[f"{class_name}_recall"] = float(recall_per_class[idx])
            metrics[f"{class_name}_f1"] = float(f1_per_class[idx])
    
    return metrics


def calculate_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: list = None
) -> np.ndarray:
    """Calculate confusion matrix."""
    return confusion_matrix(labels, predictions)


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: list = None
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device to run on
        class_names: Optional list of class names
    
    Returns:
        metrics: Dictionary of metrics
        all_predictions: All predictions
        all_labels: All labels
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            images, labels, _ = batch
            
            # Handle different input formats
            # For Qwen VLM, images should be PIL Images (list)
            if isinstance(images, list):
                images_list = images
            elif isinstance(images, torch.Tensor):
                # Convert tensor to list of PIL Images if needed
                from torchvision.transforms import ToPILImage
                to_pil = ToPILImage()
                images_list = [to_pil(img) for img in images]
            else:
                images_list = list(images)
            
            labels = labels.to(device)
            
            # Get predictions
            outputs = model(images_list)
            logits = outputs["logits"]
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    metrics = calculate_metrics(all_predictions, all_labels, class_names)
    confusion = calculate_confusion_matrix(all_predictions, all_labels, class_names)
    
    return metrics, all_predictions, all_labels, confusion
