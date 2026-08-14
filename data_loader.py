"""Data loader, deterministic split creator, and crash-resilient Incremental Logger for ColorBench."""

import os
import json
import io
import random
from PIL import Image
from typing import Dict, List, Any, Generator, Set, Tuple, Optional
from config import DATASET_NAME

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


class ColorBenchDataLoader:
    """Streams and manages ColorBench task instances from HuggingFace."""

    def __init__(self, task_filter: str = "Color Mimicry"):
        self.task_filter = task_filter
        self._dataset = None

    def _get_dataset(self):
        if not HAS_DATASETS:
            raise RuntimeError("Install datasets: pip install datasets")
        if self._dataset is None:
            print(f"[Data Loader] Loading {DATASET_NAME} split='test' from HuggingFace...")
            self._dataset = load_dataset(DATASET_NAME, split="test")
        return self._dataset

    def get_all_task_items(self) -> List[Dict[str, Any]]:
        """Return all instances belonging to this task."""
        ds = self._get_dataset()
        items = []
        for idx, row in enumerate(ds):
            row_task = row.get("task", "")
            if self.task_filter.lower() in row_task.lower():
                img = row.get("image")
                if isinstance(img, bytes):
                    img = Image.open(io.BytesIO(img)).convert("RGB")
                elif not isinstance(img, Image.Image):
                    img = Image.new("RGB", (224, 224), color="gray")

                items.append({
                    "idx": row.get("idx", idx),
                    "id": row.get("id", idx),
                    "task": row_task,
                    "type": row.get("type", ""),
                    "question": row["question"],
                    "choices": row["choices"],
                    "answer": row["answer"],
                    "image": img,
                })
        print(f"[Data Loader] Found {len(items)} instances for task filter '{self.task_filter}'.")
        return items

    def create_or_load_splits(
        self,
        num_adaptation: int = 10,
        num_eval: int = 15,
        seed: int = 42,
        splits_dir: str = "./splits",
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Create or load deterministic adaptation and held-out evaluation splits.
        Saves split metadata to JSON to guarantee identical instances across runs.
        """
        os.makedirs(splits_dir, exist_ok=True)
        clean_task = self.task_filter.lower().replace(" ", "_")
        split_file = os.path.join(splits_dir, f"{clean_task}_seed{seed}.json")

        all_items = self.get_all_task_items()
        if len(all_items) < (num_adaptation + num_eval):
            # Scale proportionally if task has fewer items
            total = len(all_items)
            num_adaptation = min(num_adaptation, max(1, total // 3))
            num_eval = total - num_adaptation

        if os.path.exists(split_file):
            print(f"[Data Loader] Loading existing split from {split_file}...")
            with open(split_file, "r", encoding="utf-8") as f:
                split_meta = json.load(f)
            adapt_indices = set(split_meta["adaptation_indices"])
            eval_indices = set(split_meta["evaluation_indices"])
            
            adapt_items = [it for it in all_items if it["idx"] in adapt_indices]
            eval_items = [it for it in all_items if it["idx"] in eval_indices]
            return adapt_items, eval_items

        print(f"[Data Loader] Creating new deterministic split (seed={seed}, adapt={num_adaptation}, eval={num_eval})...")
        rng = random.Random(seed)
        shuffled_items = list(all_items)
        rng.shuffle(shuffled_items)

        adapt_items = shuffled_items[:num_adaptation]
        eval_items = shuffled_items[num_adaptation:num_adaptation + num_eval]

        split_meta = {
            "task": self.task_filter,
            "seed": seed,
            "total_task_items": len(all_items),
            "adaptation_count": len(adapt_items),
            "evaluation_count": len(eval_items),
            "adaptation_indices": [it["idx"] for it in adapt_items],
            "evaluation_indices": [it["idx"] for it in eval_items],
        }

        with open(split_file, "w", encoding="utf-8") as f:
            json.dump(split_meta, f, indent=2)
        print(f"[Data Loader] Saved split metadata to {split_file}.")

        return adapt_items, eval_items


class IncrementalLogger:
    """Row-by-row JSONL writer with automatic resume on crash/restart."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        self.processed_indices: Set[int] = self._get_processed_indices()

    def _get_processed_indices(self) -> Set[int]:
        """Scan existing JSONL to find already-completed indices."""
        processed = set()
        if not os.path.exists(self.filepath):
            return processed
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
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
        """Append a single record immediately to disk."""
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.processed_indices.add(record["idx"])
