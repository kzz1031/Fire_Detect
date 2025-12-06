"""
Evaluate pretrained Qwen2-VL-7B on test set without fine-tuning
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict
import termcolor
import glob
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Import and configure huggingface_hub
from huggingface_hub import constants as hf_constants
hf_constants.ENDPOINT = "https://hf-mirror.com"

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
try:
    from transformers import Qwen2VLProcessor
except ImportError:
    # Fallback to AutoProcessor
    Qwen2VLProcessor = AutoProcessor
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.metrics import calculate_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Qwen2VLPretrainedEvaluator:
    """Evaluator for pretrained Qwen2-VL model without fine-tuning."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 10,
    ):
        """
        Initialize the evaluator.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on
            max_new_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        
        # Class mapping
        self.class_names = ["fire", "start_fire", "no_fire"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load model and processor
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        try:
            self.processor = Qwen2VLProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
        except:
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
        
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def load_test_data(self, test_dir: str) -> List[Dict]:
        """
        Load test dataset.
        
        Args:
            test_dir: Path to test directory
        
        Returns:
            List of samples with image path and label
        """
        test_dir = Path(test_dir)
        samples = []
        
        for class_name in self.class_names:
            class_dir = test_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory {class_dir} does not exist")
                continue
            
            # Find all image files
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            image_paths = []
            for ext in image_extensions:
                image_paths.extend(glob.glob(str(class_dir / ext)))
                image_paths.extend(glob.glob(str(class_dir / ext.upper())))
            
            class_idx = self.class_to_idx[class_name]
            for img_path in image_paths:
                samples.append({
                    "image_path": img_path,
                    "label": class_idx,
                    "label_name": class_name
                })
        
        logger.info(f"Loaded {len(samples)} test samples")
        return samples
    
    def predict_single(self, image_path: str) -> str:
        """
        Predict class for a single image.
        
        Args:
            image_path: Path to image
        
        Returns:
            Predicted class name
        """
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return "no_fire"  # Default fallback
        
        # Create prompt for classification
        prompt = (
            "Look at this image carefully. "
            "Classify it into one of these three categories: "
            "'fire', 'start_fire', or 'no_fire'. "
            "Respond with only one word: fire, start_fire, or no_fire."
        )
        
        # Process inputs using Qwen2-VL format
        try:
            # For Qwen2-VL, we need to insert image token in text
            # The processor expects text with <|image_pad|> token to mark image position
            # Get the image token from processor
            image_token = getattr(self.processor, 'image_token', '<|image_pad|>')
            
            # Insert image token at the beginning of the prompt
            # Format: <|image_pad|> + prompt text
            text_with_image = f"{image_token}\n{prompt}"
            
            # Process inputs: text contains image token, images parameter provides actual image
            inputs = self.processor(
                text=text_with_image,
                images=image,
                padding=True,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False
                )
            
            # Extract generated tokens (remove input tokens)
            input_ids = inputs["input_ids"]
            generated_ids_trimmed = generated_ids[0][input_ids.shape[1]:]
            
            # Decode response using tokenizer
            # Processor has a tokenizer attribute for decoding
            if hasattr(self.processor, 'tokenizer'):
                response = self.processor.tokenizer.decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
            else:
                # Fallback: try decode method directly
                response = self.processor.decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
            
            # Parse response to get class name
            response = response.strip().lower()
            termcolor.cprint(response, 'red')
            # Try to extract class name from response
            if "fire" in response and "start" in response:
                return "start_fire"
            elif "fire" in response and "no" not in response and "start" not in response:
                return "fire"
            elif "no" in response or "no_fire" in response:
                return "no_fire"
            else:
                # Fallback: try to find any class name in response
                for class_name in self.class_names:
                    if class_name in response:
                        return class_name
                
                logger.warning(f"Could not parse response: {response}, defaulting to no_fire")
                return "no_fire"
                
        except Exception as e:
            logger.error(f"Error during prediction for {image_path}: {e}")
            return "no_fire"
    
    def evaluate(self, test_dir: str, output_dir: str = None) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            test_dir: Path to test directory
            output_dir: Optional output directory for results
        
        Returns:
            Dictionary of metrics
        """
        # Load test data
        test_samples = self.load_test_data(test_dir)
        
        if len(test_samples) == 0:
            logger.error("No test samples found!")
            return {}
        
        # Run predictions
        logger.info("Running predictions...")
        predictions = []
        labels = []
        results = []
        
        for sample in tqdm(test_samples, desc="Evaluating"):
            pred_name = self.predict_single(sample["image_path"])
            pred_idx = self.class_to_idx.get(pred_name, 0)
            
            predictions.append(pred_idx)
            labels.append(sample["label"])
            
            results.append({
                "image_path": sample["image_path"],
                "true_label": sample["label_name"],
                "predicted_label": pred_name,
                "correct": pred_name == sample["label_name"]
            })
        
        # Calculate metrics
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        metrics = calculate_metrics(predictions, labels, self.class_names)
        confusion = confusion_matrix(labels, predictions)
        
        # Print results
        logger.info("\n" + "="*50)
        logger.info("Evaluation Results:")
        logger.info("="*50)
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        logger.info("\nPer-class metrics:")
        for class_name in self.class_names:
            logger.info(
                f"  {class_name}: "
                f"Precision={metrics.get(f'{class_name}_precision', 0):.4f}, "
                f"Recall={metrics.get(f'{class_name}_recall', 0):.4f}, "
                f"F1={metrics.get(f'{class_name}_f1', 0):.4f}"
            )
        logger.info("\nConfusion Matrix:")
        logger.info(confusion)
        logger.info("="*50)
        
        # Save results
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metrics
            metrics_path = output_dir / "pretrained_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved metrics to {metrics_path}")
            
            # Save detailed results
            results_path = output_dir / "pretrained_results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved detailed results to {results_path}")
            
            # Save confusion matrix
            confusion_path = output_dir / "pretrained_confusion_matrix.txt"
            with open(confusion_path, "w") as f:
                f.write("Confusion Matrix:\n")
                f.write(f"Rows: {self.class_names}\n")
                f.write(f"Columns: {self.class_names}\n\n")
                f.write(str(confusion))
            logger.info(f"Saved confusion matrix to {confusion_path}")
        
        return {
            "metrics": metrics,
            "confusion_matrix": confusion.tolist(),
            "results": results
        }


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate pretrained Qwen2-VL on test set")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="Model name from HuggingFace"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="data/test",
        help="Path to test directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/qwen_vlm/pretrained",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=10,
        help="Maximum tokens to generate"
    )
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = Qwen2VLPretrainedEvaluator(
        model_name=args.model_name,
        device=args.device,
        max_new_tokens=args.max_new_tokens
    )
    
    # Evaluate
    results = evaluator.evaluate(
        test_dir=args.test_dir,
        output_dir=args.output_dir
    )
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()

