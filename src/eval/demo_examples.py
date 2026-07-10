"""
demo_examples.py

Runs 5 hand-picked ColorBench samples through BOTH the zero-shot baseline
and the BullsEye pipeline, then prints a side-by-side comparison so you can
clearly see where BullsEye improves over the baseline.

Uses HuggingFace streaming so it never downloads the full 16GB dataset.
"""

import os
import sys
import torch
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from router.dispatcher import BullsEyeDispatcher

# ---------------------------------------------------------------------------
# Target tasks — one sample each. These are the 5 tasks that best showcase
# the different BullsEye branches.
# ---------------------------------------------------------------------------
TARGET_TASKS = [
    "Color Counting",    # reasoning-soluble  → Reasoning (CoT) branch
    "Color Extraction",  # perception-limited → Extraction (HSV) branch
    "Color Illusion",    # prior-override     → Suppression (grayscale) branch
    "Color Mimicry",     # prior-override     → Suppression (grayscale) branch
    "Color Robustness",  # robustness         → Normalization branch
]


def load_model_and_processor(model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
    print(f"\nLoading {model_id} in 4-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quantization_config,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    print("Model loaded.\n")
    return model, processor


def stream_samples(target_tasks, max_scan=2000):
    """
    Streams ColorBench from HuggingFace and collects the FIRST matching
    sample for each target task. Stops as soon as all 5 are found or
    max_scan items have been checked — whichever comes first.
    """
    print("Streaming ColorBench dataset (no full download)...")
    dataset = load_dataset(
        "umd-zhou-lab/ColorBench",
        split="test",
        streaming=True,
        trust_remote_code=True,
    )

    found = {}
    for i, item in enumerate(dataset):
        if i >= max_scan:
            break
        task = item.get("task", "")
        for target in target_tasks:
            if target not in found and target.lower() in task.lower():
                found[target] = item
                print(f"  Found sample for: {target}  (scanned {i+1} items)")
        if len(found) == len(target_tasks):
            break

    missing = [t for t in target_tasks if t not in found]
    if missing:
        print(f"  Warning: could not find samples for: {missing}")

    # Return in the original order
    return [found[t] for t in target_tasks if t in found]


def infer(model, processor, image, prompt, max_new_tokens=10):
    """Runs a single forward pass and returns the raw output text."""
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
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    return output_text.strip()


def parse_prediction(output_text):
    """Extracts the single letter prediction from raw model output."""
    if "Final Answer:" in output_text:
        after = output_text.split("Final Answer:")[-1].strip()
        return after[0].upper() if after else "N/A"
    return output_text[0].upper() if output_text else "N/A"


def run_demo():
    samples = stream_samples(TARGET_TASKS)
    model, processor = load_model_and_processor()
    dispatcher = BullsEyeDispatcher(taxonomy_path="./src/router/taxonomy_map.json")

    baseline_correct = 0
    bullseye_correct = 0
    total = len(samples)

    print("\n" + "=" * 65)
    print("  BULLSEYE vs BASELINE — SIDE-BY-SIDE DEMO (5 EXAMPLES)")
    print("=" * 65)

    for idx, item in enumerate(samples):
        task_name   = item.get("task", "Unknown")
        question    = item.get("question", "")
        options     = item.get("options", [])
        answer_idx  = item.get("answer")
        image       = item.get("image")

        options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        base_prompt  = f"{question}\n{options_text}\nAnswer with just the letter of the correct option."
        ground_truth = chr(65 + answer_idx) if isinstance(answer_idx, int) else str(answer_idx)

        print(f"\n--- Example {idx+1}: {task_name} ---")
        print(f"Question : {question}")
        print(f"Options  : {options_text}")
        print(f"Answer   : {ground_truth}")

        # ── Baseline (zero-shot, no intervention) ──────────────────────────
        base_output = infer(model, processor, image, base_prompt, max_new_tokens=10)
        base_pred   = parse_prediction(base_output)
        base_ok     = base_pred == ground_truth
        if base_ok:
            baseline_correct += 1

        print(f"\n  [BASELINE]")
        print(f"  Prompt    : (standard MCQ, no changes)")
        print(f"  Raw output: {base_output}")
        print(f"  Prediction: {base_pred}  →  {'✓ CORRECT' if base_ok else '✗ WRONG'}")

        # ── BullsEye (routed intervention) ─────────────────────────────────
        modified_image, modified_prompt = dispatcher.dispatch(task_name, image, base_prompt)

        # CoT responses need more tokens
        max_tok = 200 if "Let's think step by step" in modified_prompt else 10
        bull_output = infer(model, processor, modified_image, modified_prompt, max_new_tokens=max_tok)
        bull_pred   = parse_prediction(bull_output)
        bull_ok     = bull_pred == ground_truth
        if bull_ok:
            bullseye_correct += 1

        print(f"\n  [BULLSEYE]")
        print(f"  Branch    : {dispatcher.taxonomy_map.get(task_name, 'normalization')}")
        print(f"  Intervention added to prompt: {'yes' if modified_prompt != base_prompt else 'no'}")
        print(f"  Raw output: {bull_output[:300]}{'...' if len(bull_output) > 300 else ''}")
        print(f"  Prediction: {bull_pred}  →  {'✓ CORRECT' if bull_ok else '✗ WRONG'}")

        # Did BullsEye change the outcome?
        if not base_ok and bull_ok:
            print(f"\n  *** IMPROVEMENT: BullsEye fixed this one! ***")
        elif base_ok and not bull_ok:
            print(f"\n  !! REGRESSION: BullsEye broke this one.")
        else:
            print(f"\n  (No change in outcome)")

        print("-" * 65)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  SUMMARY")
    print(f"{'=' * 65}")
    print(f"  Baseline accuracy : {baseline_correct}/{total}  ({baseline_correct/total*100:.0f}%)")
    print(f"  BullsEye accuracy : {bullseye_correct}/{total}  ({bullseye_correct/total*100:.0f}%)")
    delta = bullseye_correct - baseline_correct
    sign  = "+" if delta >= 0 else ""
    print(f"  Delta             : {sign}{delta} examples  ({sign}{delta/total*100:.0f}%)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_demo()
