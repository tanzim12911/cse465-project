import os
import argparse
from datasets import load_dataset

def download_and_prepare_dataset(save_dir="./data/colorbench", max_samples=None):
    """
    Downloads the ColorBench dataset from HuggingFace and saves it locally.

    Args:
        save_dir:    Where to save the dataset on disk.
        max_samples: If set, only download this many samples (uses streaming).
                     Leave as None to download the full dataset (~16 GB).
    """
    print("Loading ColorBench dataset from HuggingFace (umd-zhou-lab/ColorBench)...")

    try:
        if max_samples is not None:
            # Streaming mode: pull only max_samples rows, then materialise to disk.
            print(f"Streaming mode: fetching first {max_samples} samples only...")
            streamed = load_dataset(
                "umd-zhou-lab/ColorBench",
                split="test",
                streaming=True,
            )
            # Convert the head of the stream to a regular Dataset
            from datasets import Dataset
            rows = []
            for i, item in enumerate(streamed):
                if i >= max_samples:
                    break
                rows.append(item)
            dataset = Dataset.from_list(rows)
            # Wrap in a DatasetDict so downstream code (load_from_disk) works the same way
            from datasets import DatasetDict
            dataset = DatasetDict({"test": dataset})
        else:
            # Full download
            dataset = load_dataset("umd-zhou-lab/ColorBench")

        print(f"Dataset loaded. Splits: {list(dataset.keys())}")
        os.makedirs(save_dir, exist_ok=True)
        dataset.save_to_disk(save_dir)
        print(f"Dataset saved to {save_dir}")
        return dataset

    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Download only this many samples (streaming). Omit for full dataset."
    )
    parser.add_argument(
        "--save_dir", type=str, default="./data/colorbench",
        help="Directory to save the dataset."
    )
    args = parser.parse_args()
    download_and_prepare_dataset(save_dir=args.save_dir, max_samples=args.max_samples)
