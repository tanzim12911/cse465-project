import os
from datasets import load_from_disk

def create_image_stripped_split(input_dir="./data/colorbench", output_dir="./data/colorbench_stripped"):
    """
    Creates a text-only variant of the ColorBench dataset by stripping the images.
    This is used for Phase 1 (Blind Baseline) to isolate language priors.
    """
    print(f"Loading dataset from {input_dir}...")
    try:
        dataset = load_from_disk(input_dir)
    except Exception as e:
        print(f"Error loading dataset: {e}. Please run download_colorbench.py first.")
        return
        
    print("Stripping images to create text-only splits...")
    
    # The dataset might be a DatasetDict. We map over it to remove the image column.
    # Alternatively, we can just remove the 'image' column.
    if 'image' in dataset.column_names or (hasattr(dataset, 'keys') and any('image' in dataset[split].column_names for split in dataset.keys())):
        stripped_dataset = dataset.remove_columns('image')
        print("Successfully removed 'image' column.")
    else:
        print("'image' column not found or already stripped.")
        stripped_dataset = dataset
        
    stripped_dataset.save_to_disk(output_dir)
    print(f"Image-stripped dataset saved to {output_dir}")

if __name__ == "__main__":
    create_image_stripped_split()
