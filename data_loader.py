"""Data loader and crash-resilient Incremental Logger for ColorBench."""

import os
import json
import io
from PIL import Image
from typing import Dict, Any, Generator, Set
from config import DATASET_NAME

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


class ColorBenchDataLoader:
    """Streams ColorBench instances row-by-row from HuggingFace."""

    def __init__(self, task_filter: str = "Color Recognition"):
        self.task_filter = task_filter

    def stream_instances(self) -> Generator[Dict[str, Any], None, None]:
        """Yield filtered dataset instances one by one."""
        if not HAS_DATASETS:
            raise RuntimeError("Install datasets: pip install datasets")

        print(f"[Data Loader] Loading {DATASET_NAME} from HuggingFace...")
        ds = load_dataset(DATASET_NAME, split="test")

        for idx, row in enumerate(ds):
            row_task = row.get("task", "")
            if self.task_filter.lower() in row_task.lower():
                img = row.get("image")
                if isinstance(img, bytes):
                    img = Image.open(io.BytesIO(img)).convert("RGB")
                elif not isinstance(img, Image.Image):
                    img = Image.new("RGB", (224, 224), color="gray")

                yield {
                    "idx": row.get("idx", idx),
                    "id": row.get("id", idx),
                    "task": row_task,
                    "type": row.get("type", ""),
                    "question": row["question"],
                    "choices": row["choices"],
                    "answer": row["answer"],
                    "image": img,
                }


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
