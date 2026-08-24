"""Dataset adapters, deterministic splits, and incremental result logging."""

import io
import json
import os
import random
from typing import Any, Dict, List, Set, Tuple

from PIL import Image

from config import COLORBENCH_DATASET, RCID_DATASET

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


class ColorIllusionDataLoader:
    """Load Color Illusion examples from ColorBench or RCID.

    RCID's public Hugging Face release is an image-folder dataset with a
    class label rather than a VQA question. The adapter turns each label into
    a binary colour-comparison question. It does not mix RCID with ColorBench:
    each source has a separate split cache and separate output name.
    """

    DATASETS = {
        "colorbench": COLORBENCH_DATASET,
        "rcid": RCID_DATASET,
    }

    def __init__(self, dataset: str = "colorbench"):
        self.dataset_key = dataset.strip().lower()
        if self.dataset_key not in self.DATASETS:
            raise ValueError(f"Unsupported dataset '{dataset}'. Choose: {', '.join(self.DATASETS)}")
        self.dataset_name = self.DATASETS[self.dataset_key]
        self.task_name = "Color Illusion" if self.dataset_key == "colorbench" else "RCID Color Illusion"
        self._dataset = None

    @property
    def output_tag(self) -> str:
        return self.dataset_key

    def _get_dataset(self):
        if not HAS_DATASETS:
            raise RuntimeError("Install datasets: pip install datasets")
        if self._dataset is None:
            print(f"[Data Loader] Loading {self.dataset_name} split='test' from HuggingFace...")
            self._dataset = load_dataset(self.dataset_name, split="test")
        return self._dataset

    @staticmethod
    def _image(row: Dict[str, Any]) -> Image.Image:
        image = row.get("image")
        if isinstance(image, bytes):
            return Image.open(io.BytesIO(image)).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        raise ValueError("Dataset row has no usable image.")

    def _colorbench_items(self, ds) -> List[Dict[str, Any]]:
        items = []
        for fallback_idx, row in enumerate(ds):
            if row.get("task", "").strip().lower() != "color illusion":
                continue
            items.append({
                "idx": row.get("idx", fallback_idx),
                "id": row.get("id", fallback_idx),
                "task": self.task_name,
                "type": row.get("type", ""),
                "question": row["question"],
                "choices": row["choices"],
                "answer": row["answer"],
                "image": self._image(row),
            })
        return items

    @staticmethod
    def _rcid_label_name(ds, label: Any) -> str:
        feature = ds.features.get("label")
        if feature is not None and hasattr(feature, "int2str"):
            return feature.int2str(int(label))
        return str(label)

    def _rcid_items(self, ds) -> List[Dict[str, Any]]:
        items = []
        for idx, row in enumerate(ds):
            label_name = self._rcid_label_name(ds, row.get("label", "")).lower()
            if "different" in label_name:
                answer = "(A)"
            elif "same" in label_name:
                answer = "(B)"
            else:
                raise ValueError(
                    f"Unsupported RCID label '{label_name}'. Expected a label containing 'same' or 'different'."
                )
            items.append({
                "idx": idx,
                "id": idx,
                "task": self.task_name,
                "type": label_name,
                "question": "Do the two target regions appear to have different colors?",
                "choices": ["Yes, they appear different", "No, they appear the same"],
                "answer": answer,
                "image": self._image(row),
            })
        return items

    def get_all_task_items(self) -> List[Dict[str, Any]]:
        ds = self._get_dataset()
        items = self._colorbench_items(ds) if self.dataset_key == "colorbench" else self._rcid_items(ds)
        print(f"[Data Loader] Found {len(items)} {self.task_name} instances.")
        return items

    def create_or_load_splits(
        self,
        num_adaptation: int = 10,
        num_eval: int = 15,
        seed: int = 42,
        splits_dir: str = "./splits",
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        os.makedirs(splits_dir, exist_ok=True)
        split_file = os.path.join(
            splits_dir, f"{self.output_tag}_color_illusion_seed{seed}_a{num_adaptation}_e{num_eval}.json"
        )
        all_items = self.get_all_task_items()
        if len(all_items) < num_adaptation + num_eval:
            num_adaptation = min(num_adaptation, max(1, len(all_items) // 3))
            num_eval = len(all_items) - num_adaptation

        if os.path.exists(split_file):
            print(f"[Data Loader] Loading existing split from {split_file}...")
            with open(split_file, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            adaptation_indices = set(metadata["adaptation_indices"])
            evaluation_indices = set(metadata["evaluation_indices"])
            return (
                [item for item in all_items if item["idx"] in adaptation_indices],
                [item for item in all_items if item["idx"] in evaluation_indices],
            )

        print(f"[Data Loader] Creating deterministic split (seed={seed}, adapt={num_adaptation}, eval={num_eval})...")
        shuffled = list(all_items)
        random.Random(seed).shuffle(shuffled)
        adaptation, evaluation = shuffled[:num_adaptation], shuffled[num_adaptation:num_adaptation + num_eval]
        with open(split_file, "w", encoding="utf-8") as handle:
            json.dump({
                "dataset": self.dataset_key,
                "task": self.task_name,
                "seed": seed,
                "total_task_items": len(all_items),
                "adaptation_count": len(adaptation),
                "evaluation_count": len(evaluation),
                "adaptation_indices": [item["idx"] for item in adaptation],
                "evaluation_indices": [item["idx"] for item in evaluation],
            }, handle, indent=2)
        print(f"[Data Loader] Saved split metadata to {split_file}.")
        return adaptation, evaluation


class IncrementalLogger:
    """Row-by-row JSONL writer with automatic resume on crash/restart."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        self.processed_indices: Set[int] = self._get_processed_indices()

    def _get_processed_indices(self) -> Set[int]:
        processed = set()
        if not os.path.exists(self.filepath):
            return processed
        with open(self.filepath, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                    if "idx" in record:
                        processed.add(record["idx"])
                except json.JSONDecodeError:
                    continue
        if processed:
            print(f"[Logger] Resuming: {len(processed)} records already completed.")
        return processed

    def log_result(self, record: Dict[str, Any]):
        with open(self.filepath, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.processed_indices.add(record["idx"])
