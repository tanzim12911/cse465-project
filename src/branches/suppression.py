import cv2
import numpy as np
from PIL import Image, ImageEnhance
import torch
from peft import LoraConfig, get_peft_model

def apply_suppression_branch(image_pil, prompt):
    """
    Applies the Suppression Branch for 'prior-override' tasks (e.g. Illusions, Mimicry).
    Converts image to grayscale to remove misleading color cues, and modifies prompt.
    """
    # Convert to grayscale
    gray_image = image_pil.convert('L')
    # Convert back to RGB format so the vision model accepts it (as a 3-channel gray image)
    gray_rgb_image = gray_image.convert('RGB')
    
    # Add pixel-level sampling prompt to enforce strict visual grounding over priors
    injection = "\n[Instruction: Ignore any textual priors about color. Rely STRICTLY on the luminance and shapes in this grayscale image to answer.]\n"
    
    if "Answer with just the letter" in prompt:
        parts = prompt.rsplit("\nAnswer with just the letter", 1)
        modified_prompt = parts[0] + injection + "\nAnswer with just the letter" + (parts[1] if len(parts) > 1 else "")
    else:
        modified_prompt = prompt + injection
        
    return gray_rgb_image, modified_prompt

def setup_lora_fine_tuning(model):
    """
    Sets up the model with LoRA for targeted fine-tuning on suppression tasks.
    (This is a helper function to be called during a separate training phase).
    """
    print("Setting up LoRA for Suppression Branch Fine-Tuning...")
    config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["q_proj", "v_proj"], # Qwen2 typically targets attention projections
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    return peft_model
