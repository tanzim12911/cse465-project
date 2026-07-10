import os
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

def load_model_and_processor(model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
    """
    Loads Qwen2.5-VL-7B in 4-bit quantization for Colab Free Tier.
    """
    print(f"Loading {model_id} in 4-bit...")
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

def run_zero_shot_evaluation(model, processor, dataset_split, num_samples=None):
    """
    Evaluates the model on the ColorBench dataset split.
    """
    correct = 0
    total = 0
    
    samples = dataset_split if num_samples is None else dataset_split.select(range(min(num_samples, len(dataset_split))))
    
    for item in tqdm(samples, desc="Evaluating"):
        image = item.get("image")
        question = item.get("question")
        options = item.get("options", [])
        answer_idx = item.get("answer") # Usually index like 0 for A, 1 for B
        
        # Format options
        options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        prompt = f"{question}\n{options_text}\nAnswer with just the letter of the correct option."
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
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
            generated_ids = model.generate(**inputs, max_new_tokens=10)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        
        # Simple heuristic to extract the predicted option (A, B, C, D, etc.)
        prediction = output_text.strip()[0].upper() if len(output_text.strip()) > 0 else "N/A"
        ground_truth = chr(65 + answer_idx) if isinstance(answer_idx, int) else answer_idx
        
        if prediction == ground_truth:
            correct += 1
        total += 1
        
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    return accuracy

if __name__ == "__main__":
    from datasets import load_from_disk
    try:
        # Assuming dataset was downloaded via download_colorbench.py
        dataset = load_from_disk("./data/colorbench")
        # Colorbench may have a 'test' split
        split_name = list(dataset.keys())[0] 
        eval_data = dataset[split_name]
    except Exception as e:
        print("Dataset not found locally, loading from HuggingFace...")
        dataset = load_dataset("umd-zhou-lab/ColorBench")
        split_name = list(dataset.keys())[0]
        eval_data = dataset[split_name]
        
    model, processor = load_model_and_processor()
    run_zero_shot_evaluation(model, processor, eval_data, num_samples=100)
