import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from datasets import load_from_disk


class FeatureExtractor:
    """
    Attaches forward hooks to three checkpoints inside the model:
      - vision_encoder : output of the visual tower
      - llm_mid        : middle transformer layer
      - llm_out        : final transformer layer
    """

    def __init__(self, model):
        self.model = model
        self.features = {}
        self.hooks = []
        self._register_hooks()

    def _get_hook(self, name):
        def hook(module, input, output):
            # output can be a tuple (hidden_state, ...) or a plain tensor
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output
            # float() converts bfloat16/float16 → float32 so numpy works cleanly
            self.features[name] = tensor.detach().float().cpu()
        return hook

    def _register_hooks(self):
        registered = []
        try:
            self.hooks.append(
                self.model.visual.register_forward_hook(self._get_hook("vision_encoder"))
            )
            registered.append("vision_encoder")
        except AttributeError:
            pass

        try:
            mid = len(self.model.model.layers) // 2
            self.hooks.append(
                self.model.model.layers[mid].register_forward_hook(self._get_hook("llm_mid"))
            )
            registered.append("llm_mid")
        except AttributeError:
            pass

        try:
            self.hooks.append(
                self.model.model.layers[-1].register_forward_hook(self._get_hook("llm_out"))
            )
            registered.append("llm_out")
        except AttributeError:
            pass

        print(f"Hooks registered for: {registered}")

    def clear(self):
        self.features = {}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


def _pool_feature(tensor):
    """Mean-pools a feature tensor to a 1-D numpy vector."""
    # (batch, seq, hidden) → (hidden,)
    if tensor.dim() == 3:
        return tensor.mean(dim=1).squeeze(0).numpy()
    # (seq, hidden) → (hidden,)
    if tensor.dim() == 2:
        return tensor.mean(dim=0).numpy()
    # already 1-D
    return tensor.numpy()


def extract_and_save_features(model_id, dataset_path="./data/colorbench",
                               output_dir="./data/features", max_samples=200,
                               model=None, processor=None):
    """
    Runs forward passes and saves hidden-state features to disk.
    Pass model/processor to reuse an already-loaded model from the notebook.
    """
    if model is None or processor is None:
        print(f"Loading {model_id} in 4-bit...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        )
        processor = AutoProcessor.from_pretrained(model_id)
    else:
        print("Reusing already-loaded model for feature extraction.")

    extractor = FeatureExtractor(model)

    dataset   = load_from_disk(dataset_path)
    split     = list(dataset.keys())[0]
    n         = min(max_samples, len(dataset[split]))
    eval_data = dataset[split].select(range(n))
    print(f"Extracting features from {n} samples...")

    all_features = {k: [] for k in ["vision_encoder", "llm_mid", "llm_out", "labels", "task"]}

    for item in tqdm(eval_data, desc="Extracting Features"):
        image      = item.get("image")
        question   = item.get("question", "")
        task_name  = item.get("task", "Unknown")
        answer_raw = item.get("answer")

        # Normalise answer to a single letter string
        if isinstance(answer_raw, int):
            label = chr(65 + answer_raw)
        else:
            s = str(answer_raw).strip().upper()
            label = next((c for c in s if c.isalpha()), s)

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": question},
        ]}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(model.device)

        extractor.clear()
        with torch.no_grad():
            model(**inputs)

        for key in ["vision_encoder", "llm_mid", "llm_out"]:
            feat = extractor.features.get(key)
            if feat is not None:
                all_features[key].append(_pool_feature(feat))
            else:
                # placeholder so list lengths stay aligned
                all_features[key].append(None)

        all_features["labels"].append(label)
        all_features["task"].append(task_name)

    extractor.remove_hooks()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "features.pt")
    torch.save(all_features, out_path)
    print(f"Features saved to {out_path}  ({n} samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",     type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset_path", type=str, default="./data/colorbench")
    parser.add_argument("--output_dir",   type=str, default="./data/features")
    parser.add_argument("--max_samples",  type=int, default=200)
    args = parser.parse_args()
    extract_and_save_features(
        model_id=args.model_id,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )
