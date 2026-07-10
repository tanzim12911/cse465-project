"""
custom_demo.py

Demo using YOUR OWN uploaded images instead of the dataset.
Upload 5 images to Colab (or Drive), fill in the CUSTOM_SAMPLES list below,
then run this script.

Each sample needs:
  - image_path : path to the image file
  - task       : one of the 11 ColorBench task names (controls which branch runs)
  - question   : the question to ask the model
  - options    : list of answer strings  ["red", "blue", "green", "yellow"]
  - answer     : index of the correct option (0=A, 1=B, 2=C, 3=D)

Branch routing by task name:
  reasoning-soluble  → Color Recognition, Color Proportion, Color Comparison,
                        Color Counting, Object Counting, Color Blindness, Object Recognition
  perception-limited → Color Extraction
  prior-override     → Color Illusion, Color Mimicry
  normalization      → Color Robustness
"""

import os
import sys
import argparse
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from router.dispatcher import BullsEyeDispatcher

# ---------------------------------------------------------------------------
# EDIT THIS SECTION — add your 5 images here
# ---------------------------------------------------------------------------
CUSTOM_SAMPLES = [
    {
        "image_path": "./demo_images/image1.jpg",
        "task":       "Color Counting",        # → Reasoning branch (CoT)
        "question":   "How many distinct colors of shirts are in this image?",
        "options":    ["2", "3", "4", "5"],
        "answer":     2,                        # 0=A, 1=B, 2=C, 3=D  →  C = "4"
    },
    {
        "image_path": "./demo_images/image2.jpg",
        "task":       "Color Extraction",       # → Extraction branch (HSV)
        "question":   "What is the HSV value of the highlighted color?",
        "options":    ["[120, 50, 80]", "[180, 70, 60]", "[200, 80, 50]", "[240, 90, 40]"],
        "answer":     1,
    },
    {
        "image_path": "./demo_images/image3.jpg",
        "task":       "Color Illusion",         # → Suppression branch (grayscale)
        "question":   "Do the two marked squares have the same color?",
        "options":    ["Yes", "No, left is darker", "No, right is darker", "Cannot tell"],
        "answer":     0,
    },
    {
        "image_path": "./demo_images/image4.jpg",
        "task":       "Color Mimicry",          # → Suppression branch (grayscale)
        "question":   "How many red objects are camouflaged in this image?",
        "options":    ["0", "1", "2", "3"],
        "answer":     0,
    },
    {
        "image_path": "./demo_images/image5.jpg",
        "task":       "Color Robustness",       # → Normalization branch
        "question":   "What color is the car in this image?",
        "options":    ["Red", "Blue", "Green", "Yellow"],
        "answer":     1,
    },
]
# ---------------------------------------------------------------------------


def load_model_and_processor(model_id):
    print(f"Loading {model_id} in 4-bit...")
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
    print("Model loaded.\n")
    return model, processor


def infer(model, processor, image_pil, prompt, max_new_tokens=10):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_pil},
        {"type": "text",  "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return processor.batch_decode(trimmed, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)[0].strip()


def parse_prediction(output_text):
    if "Final Answer:" in output_text:
        after = output_text.split("Final Answer:")[-1].strip()
        return after[0].upper() if after else "N/A"
    return output_text[0].upper() if output_text else "N/A"


def run_custom_demo(model_id, output_dir):
    model, processor = load_model_and_processor(model_id)
    dispatcher = BullsEyeDispatcher(taxonomy_path="./src/router/taxonomy_map.json")

    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "custom_demo_results.txt")

    baseline_correct = 0
    bullseye_correct = 0
    total = len(CUSTOM_SAMPLES)
    lines = []

    header = "=" * 65
    lines.append(header)
    lines.append("  BULLSEYE vs BASELINE — CUSTOM IMAGE DEMO")
    lines.append(f"  Model: {model_id}")
    lines.append(header)

    for idx, sample in enumerate(CUSTOM_SAMPLES):
        image_path = sample["image_path"]
        task_name  = sample["task"]
        question   = sample["question"]
        options    = sample["options"]
        answer_idx = sample["answer"]

        if not os.path.exists(image_path):
            msg = f"[Example {idx+1}] Image not found: {image_path} — skipping."
            print(msg)
            lines.append(msg)
            continue

        image = Image.open(image_path).convert("RGB")
        options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        base_prompt  = f"{question}\n{options_text}\nAnswer with just the letter of the correct option."
        ground_truth = chr(65 + answer_idx)

        block = [
            f"\n--- Example {idx+1}: {task_name} ---",
            f"Image    : {image_path}",
            f"Question : {question}",
            f"Options  : {options_text}",
            f"Answer   : {ground_truth}",
        ]

        # ── Baseline ────────────────────────────────────────────────────────
        base_out  = infer(model, processor, image, base_prompt, max_new_tokens=10)
        base_pred = parse_prediction(base_out)
        base_ok   = base_pred == ground_truth
        if base_ok:
            baseline_correct += 1

        block += [
            "",
            "  [BASELINE — zero-shot, no intervention]",
            f"  Raw output : {base_out}",
            f"  Prediction : {base_pred}  →  {'✓ CORRECT' if base_ok else '✗ WRONG'}",
        ]

        # ── BullsEye ────────────────────────────────────────────────────────
        modified_image, modified_prompt = dispatcher.dispatch(task_name, image, base_prompt)
        max_tok  = 200 if "Let's think step by step" in modified_prompt else 10
        bull_out  = infer(model, processor, modified_image, modified_prompt, max_new_tokens=max_tok)
        bull_pred = parse_prediction(bull_out)
        bull_ok   = bull_pred == ground_truth
        if bull_ok:
            bullseye_correct += 1

        branch = dispatcher.taxonomy_map.get(task_name, "normalization")
        block += [
            "",
            f"  [BULLSEYE — branch: {branch}]",
            f"  Raw output : {bull_out[:300]}{'...' if len(bull_out) > 300 else ''}",
            f"  Prediction : {bull_pred}  →  {'✓ CORRECT' if bull_ok else '✗ WRONG'}",
        ]

        if not base_ok and bull_ok:
            block.append("  *** IMPROVEMENT: BullsEye fixed this one! ***")
        elif base_ok and not bull_ok:
            block.append("  !! REGRESSION: BullsEye broke this one.")
        else:
            block.append("  (No change in outcome)")

        block.append("-" * 65)

        for line in block:
            print(line)
        lines.extend(block)

    # ── Summary ─────────────────────────────────────────────────────────────
    delta = bullseye_correct - baseline_correct
    sign  = "+" if delta >= 0 else ""
    summary = [
        "",
        "=" * 65,
        "  SUMMARY",
        "=" * 65,
        f"  Model             : {model_id}",
        f"  Baseline accuracy : {baseline_correct}/{total}  ({baseline_correct/total*100:.0f}%)",
        f"  BullsEye accuracy : {bullseye_correct}/{total}  ({bullseye_correct/total*100:.0f}%)",
        f"  Delta             : {sign}{delta} examples  ({sign}{delta/total*100:.0f}%)",
        "=" * 65,
    ]
    for line in summary:
        print(line)
    lines.extend(summary)

    # Save to Drive / output_dir
    with open(results_file, "w") as f:
        f.write("\n".join(lines))
    print(f"\nFull results saved to: {results_file}")


def run_custom_demo_inline(model, processor, output_dir):
    """
    Called directly from the notebook with an already-loaded model.
    No weight reload — reuses whatever is in GPU memory.
    """
    dispatcher = BullsEyeDispatcher()
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "custom_demo_results.txt")

    baseline_correct = 0
    bullseye_correct = 0
    total = len(CUSTOM_SAMPLES)
    lines = []

    sep = "=" * 65
    lines += [sep, "  BULLSEYE vs BASELINE — CUSTOM IMAGE DEMO",
              f"  Model: {model.config._name_or_path}", sep]

    for idx, sample in enumerate(CUSTOM_SAMPLES):
        image_path = sample["image_path"]
        task_name  = sample["task"]
        question   = sample["question"]
        options    = sample["options"]
        answer_idx = sample["answer"]

        if not os.path.exists(image_path):
            msg = f"[Example {idx+1}] Image not found: {image_path} — skipping."
            print(msg); lines.append(msg); continue

        image        = Image.open(image_path).convert("RGB")
        options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        base_prompt  = f"{question}\n{options_text}\nAnswer with just the letter of the correct option."
        ground_truth = chr(65 + answer_idx)

        block = [f"\n--- Example {idx+1}: {task_name} ---",
                 f"Question : {question}", f"Options  :\n{options_text}",
                 f"Answer   : {ground_truth}"]

        base_out  = infer(model, processor, image, base_prompt, 10)
        base_pred = parse_prediction(base_out)
        base_ok   = base_pred == ground_truth
        if base_ok: baseline_correct += 1

        block += ["", "  [BASELINE — zero-shot, no intervention]",
                  f"  Raw output : {base_out}",
                  f"  Prediction : {base_pred}  →  {'✓ CORRECT' if base_ok else '✗ WRONG'}"]

        modified_image, modified_prompt = dispatcher.dispatch(task_name, image, base_prompt)
        max_tok  = 200 if "step by step" in modified_prompt else 10
        bull_out  = infer(model, processor, modified_image, modified_prompt, max_tok)
        bull_pred = parse_prediction(bull_out)
        bull_ok   = bull_pred == ground_truth
        if bull_ok: bullseye_correct += 1

        branch = dispatcher.taxonomy_map.get(task_name, "normalization")
        block += ["", f"  [BULLSEYE — branch: {branch}]",
                  f"  Raw output : {bull_out[:300]}{'...' if len(bull_out)>300 else ''}",
                  f"  Prediction : {bull_pred}  →  {'✓ CORRECT' if bull_ok else '✗ WRONG'}"]

        if not base_ok and bull_ok:
            block.append("  *** IMPROVEMENT: BullsEye fixed this one! ***")
        elif base_ok and not bull_ok:
            block.append("  !! REGRESSION: BullsEye broke this one.")
        else:
            block.append("  (No change in outcome)")

        block.append("-" * 65)
        for line in block: print(line)
        lines.extend(block)

    delta = bullseye_correct - baseline_correct
    sign  = "+" if delta >= 0 else ""
    summary = ["", sep, "  SUMMARY", sep,
               f"  Baseline : {baseline_correct}/{total}  ({baseline_correct/total*100:.0f}%)",
               f"  BullsEye : {bullseye_correct}/{total}  ({bullseye_correct/total*100:.0f}%)",
               f"  Delta    : {sign}{delta}  ({sign}{delta/total*100:.0f}%)", sep]
    for line in summary: print(line)
    lines.extend(summary)

    with open(results_file, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",   type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./data")
    args = parser.parse_args()
    run_custom_demo(args.model_id, args.output_dir)
