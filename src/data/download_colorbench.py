import os
from datasets import load_dataset
from PIL import Image

def download_and_prepare_dataset(save_dir="./data/colorbench"):
    """
    Downloads the ColorBench dataset from HuggingFace and saves it locally.
    """
    print("Loading ColorBench dataset from HuggingFace (umd-zhou-lab/ColorBench)...")
    
    # Attempt to load the dataset. According to Hugging Face, it might have multiple subsets or just one main one.
    # Usually it's load_dataset("umd-zhou-lab/ColorBench", split="test") since it's an evaluation benchmark.
    try:
        dataset = load_dataset("umd-zhou-lab/ColorBench")
        print(f"Dataset loaded successfully. Splits available: {dataset.keys()}")
        
        # Save to disk for easier local access in Colab
        dataset.save_to_disk(save_dir)
        print(f"Dataset saved to {save_dir}")
        return dataset
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None

if __name__ == "__main__":
    download_and_prepare_dataset()
