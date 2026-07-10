import os
import argparse
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import json


def parse_answer(answer_field):
    if isinstance(answer_field, int):
        return chr(65 + answer_field)
    s = str(answer_field).strip().upper()
    for ch in s:
        if ch.isalpha():
            return ch
    return s


def load_model_and_processor(model_id):
    print(f"Loading {model_id} in 4-bit for blind evaluation...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, device_map="auto", quantization_config=quantization_config,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def run_blind_evaluation(model, processor, dataset_split, output_dir, num_samples=None):
    """
    Text-only inference — image is completely omitted.
    Isolates how much accuracy comes from language priors alone.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "blind_results.jsonl")

    processed_ids = set()
    correct = 0
    total = 0
    task_stats = {}

    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                processed_ids.add(data.get('id'))
                task = data.get('task', 'Unknown')
                task_stats.setdefault(task, {'correct': 0, 'total': 0})
                task_stats[task]['total'] += 1
                if data.get('correct'):
                    task_stats[task]['correct'] += 1
                    correct += 1
                total += 1
        print(f"Resuming from checkpoint — {total} already processed.")

    samples = dataset_split if num_samples is None else dataset_split.select(range(min(num_samples, len(dataset_split))))

    for idx, item in enumerate(tqdm(samples, desc="Blind Evaluating")):
        item_id    = str(item.get("id", item.get("question_id", idx)))
        if item_id in processed_ids:
            continue

        question   = item.get("question")
        options    = item.get("options", [])
        answer_idx = item.get("answer")
        task_name  = item.get("task", "Unknown")

        options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        prompt = (
            f"Without seeing any image, answer the following question "
            f"to the best of your ability:\n\n{question}\n{options_text}\n"
            f"Answer with just the letter of the correct option."
        )

        # No image block — text only
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=10)
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(trimmed, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)[0]

        prediction   = output_text.strip()[0].upper() if output_text.strip() else "N/A"
        ground_truth = parse_answer(answer_idx)
        is_correct   = (prediction == ground_truth)

        task_stats.setdefault(task_name, {'correct': 0, 'total': 0})
        task_stats[task_name]['total'] += 1
        if is_correct:
            task_stats[task_name]['correct'] += 1
            correct += 1
        total += 1

        with open(output_file, 'a') as f:
            f.write(json.dumps({
                "id": item_id, "task": task_name,
                "prediction": prediction, "ground_truth": ground_truth,
                "correct": is_correct
            }) + "\n")

    print("\n--- Blind Per-Task Accuracy ---")
    for task, stats in sorted(task_stats.items()):
        acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {task:<25}: {acc:.1f}%  ({stats['correct']}/{stats['total']})")

    overall = correct / total * 100 if total > 0 else 0
    print(f"\nOverall Blind Accuracy: {overall:.2f}%  ({correct}/{total})")
    print(f"Results saved to: {output_file}")
    return overall, task_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",    type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output_dir",  type=str, default="./data")
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()

    from datasets import load_from_disk
    try:
        dataset   = load_from_disk("./data/colorbench_stripped")
        eval_data = dataset[list(dataset.keys())[0]]
    except Exception:
        print("Stripped dataset not found. Please run src/data/image_stripper.py first.")
        exit(1)

    model, processor = load_model_and_processor(args.model_id)
    run_blind_evaluation(model, processor, eval_data, args.output_dir, args.num_samples)
