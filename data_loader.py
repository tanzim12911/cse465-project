"""Data loader, dataset partitioner, and crash-resilient Incremental Logger for ColorBench."""

import os
import json
import io
from PIL import Image
from typing import Dict, Any, Generator, Set, List, Tuple
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


def partition_task_dataset(
    dataset_items: List[Dict[str, Any]],
    verifier_size: int = 5,
    oracle_size: int = 5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Deterministically partitions a task dataset into:
      1. Verifier Private Pool (verifier_size real instances, default 5)
      2. Oracle Hidden Pool (oracle_size real instances, default 5)
      3. Evaluation Stream (remaining instances >= 1)

    Raises ValueError if total instances < verifier_size + oracle_size + 1 (i.e. < 11).
    """
    min_required = verifier_size + oracle_size + 1
    total_count = len(dataset_items)
    if total_count < min_required:
        raise ValueError(
            f"Insufficient dataset instances for GVO partitioning: "
            f"Found {total_count} items, but require at least {min_required} "
            f"({verifier_size} Verifier + {oracle_size} Oracle + >=1 Evaluation)."
        )

    verifier_pool = dataset_items[:verifier_size]
    oracle_suite = dataset_items[verifier_size : verifier_size + oracle_size]
    eval_instances = dataset_items[verifier_size + oracle_size :]

    return verifier_pool, oracle_suite, eval_instances


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
