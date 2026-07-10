import os
import sys
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from datasets import load_from_disk

# Add src to path to import router
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from router.dispatcher import BullsEyeDispatcher

def load_model_and_processor(model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
    """
    Loads Qwen2.5-VL-7B in 4-bit quantization.
    """
    print(f"Loading {model_id} in 4-bit for BullsEye evaluation...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quantization_config,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def run_bullseye_evaluation(model, processor, dataset_split, num_samples=None):
    """
    Evaluates the model on the ColorBench dataset using the BullsEye Routing Architecture.
    """
    dispatcher = BullsEyeDispatcher(taxonomy_path="./src/router/taxonomy_map.json")
    
    correct = 0
    total = 0
    
    samples = dataset_split if num_samples is None else dataset_split.select(range(min(num_samples, len(dataset_split))))
    
    for item in tqdm(samples, desc="BullsEye Evaluating"):
        original_image = item.get("image")
        question = item.get("question")
        options = item.get("options", [])
        answer_idx = item.get("answer")
        task_name = item.get("task", "Unknown")
        
        options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        prompt = f"{question}\n{options_text}\nAnswer with just the letter of the correct option."
        
        # --- ROUTING INTERVENTION ---
        modified_image, modified_prompt = dispatcher.dispatch(task_name, original_image, prompt)
        # ----------------------------
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": modified_image},
                    {"type": "text", "text": modified_prompt},
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
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=150) # Increased for CoT
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        
        # Heuristic parsing since CoT might output long strings
        prediction = "N/A"
        if "Final Answer:" in output_text:
            parts = output_text.split("Final Answer:")
            if len(parts) > 1 and len(parts[1].strip()) > 0:
                prediction = parts[1].strip()[0].upper()
        else:
            prediction = output_text.strip()[0].upper() if len(output_text.strip()) > 0 else "N/A"
            
        ground_truth = chr(65 + answer_idx) if isinstance(answer_idx, int) else answer_idx
        
        if prediction == ground_truth:
            correct += 1
        total += 1
        
    accuracy = correct / total if total > 0 else 0
    print(f"BullsEye Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    return accuracy

if __name__ == "__main__":
    try:
        dataset = load_from_disk("./data/colorbench")
        split_name = list(dataset.keys())[0] 
        eval_data = dataset[split_name]
    except Exception as e:
        print("Dataset not found locally. Please run download_colorbench.py")
        exit(1)
        
    model, processor = load_model_and_processor()
    run_bullseye_evaluation(model, processor, eval_data, num_samples=100)
