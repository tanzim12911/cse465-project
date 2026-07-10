import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from datasets import load_from_disk

def get_target_modules(model):
    """
    Identifies the modules to hook into for Qwen2.5-VL-7B.
    """
    # These are heuristic paths for Qwen2.5-VL-7B. 
    # Vision encoder is typically under `visual`. Projector is also within `visual`.
    # LLM layers are under `model.layers`.
    
    # We will try to hook into the last layer of the vision encoder, the projector output, 
    # and a middle/last layer of the LLM.
    modules = {}
    
    try:
        # Qwen2.5-VL vision model
        modules['vision_encoder'] = model.visual
        # Note: The visual model includes the projector. 
        # A more precise hook might be needed for the exact encoder output vs projector output
        # For simplicity, we can hook the visual model output (which is projector output)
        # To get the raw vision encoder, we might need to hook model.visual.blocks[-1]
    except AttributeError:
        pass
        
    try:
        modules['llm_mid'] = model.model.layers[len(model.model.layers) // 2]
        modules['llm_out'] = model.model.layers[-1]
    except AttributeError:
        pass
        
    return modules

class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = {}
        self.hooks = []
        self._register_hooks()
        
    def _get_hook(self, name):
        def hook(module, input, output):
            # Output might be a tuple or tensor. We extract the tensor.
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output
                
            # Detach and move to CPU to save GPU memory
            self.features[name] = tensor.detach().cpu()
        return hook
        
    def _register_hooks(self):
        modules = get_target_modules(self.model)
        for name, module in modules.items():
            print(f"Registering hook for {name}")
            self.hooks.append(module.register_forward_hook(self._get_hook(name)))
            
    def clear(self):
        self.features = {}
        
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

def extract_and_save_features(model_id="Qwen/Qwen2.5-VL-7B-Instruct", dataset_path="./data/colorbench", output_dir="./data/features"):
    print("Loading model in 4-bit...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quantization_config)
    processor = AutoProcessor.from_pretrained(model_id)
    
    extractor = FeatureExtractor(model)
    
    dataset = load_from_disk(dataset_path)
    split_name = list(dataset.keys())[0]
    eval_data = dataset[split_name]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # We will save features in a dictionary format
    all_features = {'vision_encoder': [], 'llm_mid': [], 'llm_out': [], 'labels': [], 'task': []}
    
    # Process a subset to save time/space during prototyping
    subset = eval_data.select(range(min(200, len(eval_data))))
    
    for idx, item in enumerate(tqdm(subset, desc="Extracting Features")):
        image = item.get("image")
        question = item.get("question")
        task_name = item.get("task", "Unknown") # E.g., 'Color Recognition'
        answer_idx = item.get("answer")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        
        extractor.clear()
        
        with torch.no_grad():
            _ = model(**inputs)
            
        # Aggregate features (e.g., take the mean over the sequence length)
        for key in ['vision_encoder', 'llm_mid', 'llm_out']:
            if key in extractor.features:
                feat = extractor.features[key]
                # If feature is 3D (batch, seq, hidden), average over seq
                if feat.dim() == 3:
                    feat = feat.mean(dim=1).squeeze(0).numpy()
                elif feat.dim() == 2:
                    feat = feat.mean(dim=0).numpy()
                all_features[key].append(feat)
                
        all_features['labels'].append(answer_idx)
        all_features['task'].append(task_name)
        
    extractor.remove_hooks()
    
    print(f"Saving extracted features to {output_dir}/features.pt")
    torch.save(all_features, os.path.join(output_dir, "features.pt"))

if __name__ == "__main__":
    extract_and_save_features()
