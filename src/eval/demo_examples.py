import os
import sys
import torch
import random
from datasets import load_from_disk, load_dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from router.dispatcher import BullsEyeDispatcher

def run_demo():
    print("Loading Dataset...")
    try:
        dataset = load_from_disk("./data/colorbench")
        eval_data = dataset[list(dataset.keys())[0]]
    except Exception as e:
        print("Dataset not found locally, downloading from HuggingFace...")
        dataset = load_dataset("umd-zhou-lab/ColorBench")
        eval_data = dataset[list(dataset.keys())[0]]
    
    print("Loading Model in 4-bit...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        device_map="auto",
        quantization_config=quantization_config,
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    
    dispatcher = BullsEyeDispatcher(taxonomy_path="./src/router/taxonomy_map.json")
    
    # Let's pick 5 specific tasks to demonstrate
    target_tasks = ["Color Counting", "Color Extraction", "Color Mimicry", "Color Illusion", "Color Robustness"]
    
    selected_samples = []
    # Find one sample for each target task
    for target in target_tasks:
        for item in eval_data:
            if target.lower() in item.get('task', '').lower():
                selected_samples.append(item)
                break
                
    print("\n" + "="*50)
    print("BULLSEYE PIPELINE SIMULATION (5 EXAMPLES)")
    print("="*50 + "\n")
    
    for idx, item in enumerate(selected_samples):
        task_name = item.get("task", "Unknown")
        question = item.get("question")
        options = item.get("options", [])
        answer_idx = item.get("answer")
        ground_truth = chr(65 + answer_idx) if isinstance(answer_idx, int) else answer_idx
        
        options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        prompt = f"{question}\n{options_text}\nAnswer with just the letter of the correct option."
        
        print(f"--- Example {idx+1}: {task_name} ---")
        print(f"Original Prompt: \n{prompt}\n")
        
        # Route through BullsEye
        modified_image, modified_prompt = dispatcher.dispatch(task_name, item.get("image"), prompt)
        
        print(f"BullsEye Intervention Prompt: \n{modified_prompt}\n")
        
        # Run through model
        messages = [
            {"role": "user", "content": [{"type": "image", "image": modified_image}, {"type": "text", "text": modified_prompt}]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=150)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        print(f"Model Output:\n{output_text.strip()}\n")
        print(f"Ground Truth: {ground_truth}")
        print("="*50 + "\n")

if __name__ == "__main__":
    run_demo()
