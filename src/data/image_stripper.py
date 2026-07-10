import os
from datasets import load_from_disk, DatasetDict


def create_image_stripped_split(
    input_dir="./data/colorbench",
    output_dir="./data/colorbench_stripped"
):
    """
    Creates a text-only variant of ColorBench by removing the image column.
    Used for Phase 1 (Blind Baseline) to isolate language priors.
    """
    print(f"Loading dataset from {input_dir}...")
    try:
        dataset = load_from_disk(input_dir)
    except Exception as e:
        print(f"Error loading dataset: {e}. Please run download_colorbench.py first.")
        return

    # dataset is always a DatasetDict after load_from_disk
    if not isinstance(dataset, DatasetDict):
        print("Unexpected dataset format.")
        return

    stripped_splits = {}
    for split_name, split_data in dataset.items():
        if "image" in split_data.column_names:
            stripped_splits[split_name] = split_data.remove_columns(["image"])
            print(f"  [{split_name}] Removed 'image' column. "
                  f"Remaining columns: {stripped_splits[split_name].column_names}")
        else:
            stripped_splits[split_name] = split_data
            print(f"  [{split_name}] No 'image' column found, keeping as-is.")

    stripped_dataset = DatasetDict(stripped_splits)
    os.makedirs(output_dir, exist_ok=True)
    stripped_dataset.save_to_disk(output_dir)
    print(f"Image-stripped dataset saved to {output_dir}")


if __name__ == "__main__":
    create_image_stripped_split()
