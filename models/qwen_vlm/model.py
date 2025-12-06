"""
Qwen VLM model for fire detection classification
"""
import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
import logging

logger = logging.getLogger(__name__)


class QwenVLMForFireDetection(nn.Module):
    """
    Qwen VLM model fine-tuned for fire detection classification.
    Uses a classification head on top of the vision-language model.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen-VL",
        num_classes: int = 3,
        max_length: int = 512,
        image_size: int = 448,
        freeze_vision_encoder: bool = False,
        freeze_language_model: bool = False,
        hidden_size: int = 1024,
        dropout: float = 0.1,
        use_peft: bool = True,
        peft_type: str = "lora",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: list = None,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.max_length = max_length
        self.image_size = image_size
        self.use_peft = use_peft
        
        # Load Qwen VLM model and processor
        logger.info(f"Loading Qwen VLM model: {model_name}")
        try:
            # Try Qwen2VL first (newer version)
            self.base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"Failed to load Qwen2VL: {e}, trying Qwen-VL")
            # Fallback to Qwen-VL (older version)
            from transformers import QwenVLForConditionalGeneration
            self.base_model = QwenVLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Freeze components if needed
        if freeze_vision_encoder:
            for param in self.base_model.visual.parameters():
                param.requires_grad = False
            logger.info("Vision encoder frozen")
            
        if freeze_language_model:
            for param in self.base_model.language_model.parameters():
                param.requires_grad = False
            logger.info("Language model frozen")
        
        # Apply PEFT if enabled
        if use_peft:
            if target_modules is None:
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
            )
            self.base_model = get_peft_model(self.base_model, peft_config)
            logger.info(f"Applied LoRA with r={lora_r}, alpha={lora_alpha}")
        
        # Get hidden size from model
        if hasattr(self.base_model, 'config'):
            model_hidden_size = getattr(
                self.base_model.config,
                'hidden_size',
                getattr(
                    self.base_model.config,
                    'd_model',
                    getattr(
                        self.base_model.config,
                        'text_config',
                        type('obj', (object,), {'hidden_size': hidden_size})()
                    ).hidden_size if hasattr(self.base_model.config, 'text_config') else hidden_size
                )
            )
        else:
            model_hidden_size = hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(model_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        logger.info(f"Initialized Qwen VLM for fire detection with {num_classes} classes")
    
    def forward(self, images, texts=None, labels=None):
        """
        Forward pass for classification.
        
        Args:
            images: List of PIL Images
            texts: Optional list of text prompts (for VLM)
            labels: Optional tensor of class labels
        
        Returns:
            logits: Classification logits
            loss: Optional loss if labels provided
        """
        # Ensure images is a list
        if not isinstance(images, list):
            images = [images]
        
        # Prepare inputs
        if texts is None:
            # Default prompt for fire detection
            texts = [
                "Classify this image as: fire, start_fire, or no_fire."
            ] * len(images)
        
        # Process inputs with processor
        try:
            inputs = self.processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True
            )
        except Exception as e:
            logger.error(f"Error processing inputs: {e}")
            # Fallback: try with single image
            if len(images) == 1:
                inputs = self.processor(
                    text=texts[0] if isinstance(texts, list) else texts,
                    images=images[0],
                    return_tensors="pt",
                    padding=True
                )
            else:
                raise
        
        # Move inputs to device
        device = next(self.base_model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Get model outputs - use generate or forward
        # For classification, we need to extract features
        try:
            # Try to get hidden states from the model
            with torch.no_grad() if labels is None else torch.enable_grad():
                outputs = self.base_model(**inputs, output_hidden_states=True)
        except Exception as e:
            logger.warning(f"Error getting hidden states: {e}, trying alternative method")
            # Alternative: use the model's forward pass
            outputs = self.base_model(**inputs)
        
        # Extract features for classification
        # Qwen VLM structure may vary, try different approaches
        pooled_output = None
        
        # Method 1: Try to get from hidden_states
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            hidden_states = outputs.hidden_states[-1]
            if hidden_states.dim() == 3:
                pooled_output = hidden_states[:, 0, :]  # First token
            else:
                pooled_output = hidden_states.mean(dim=1)
        
        # Method 2: Try to get from last_hidden_state
        elif hasattr(outputs, 'last_hidden_state'):
            last_hidden = outputs.last_hidden_state
            if last_hidden.dim() == 3:
                pooled_output = last_hidden[:, 0, :]
            else:
                pooled_output = last_hidden.mean(dim=1)
        
        # Method 3: Use logits and pool
        elif hasattr(outputs, 'logits'):
            logits = outputs.logits
            if logits.dim() == 3:
                pooled_output = logits.mean(dim=1)
            else:
                pooled_output = logits
        
        # Method 4: Extract from model's language model output
        else:
            # Try to access the language model's output
            if hasattr(self.base_model, 'language_model'):
                try:
                    lm_outputs = self.base_model.language_model(**inputs)
                    if hasattr(lm_outputs, 'last_hidden_state'):
                        pooled_output = lm_outputs.last_hidden_state[:, 0, :]
                except:
                    pass
        
        if pooled_output is None:
            # Final fallback: use a dummy feature vector
            logger.warning("Could not extract features, using fallback")
            batch_size = len(images)
            hidden_size = getattr(self.base_model.config, 'hidden_size', 1024)
            pooled_output = torch.zeros(batch_size, hidden_size, device=device)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            labels = labels.to(device)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": pooled_output
        }
    
    def predict(self, images, texts=None):
        """Predict class for given images."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(images, texts)
            probs = torch.softmax(outputs["logits"], dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds, probs

