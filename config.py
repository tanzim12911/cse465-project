"""Configuration for CSE465 ColorBench Adaptive Skill Generation Pipeline (ACE Framework)."""

import os

# ==============================================================================
# Model Configurations (Unified Local Self-Improving Architecture)
# ==============================================================================
# Qwen2.5-VL-7B-Instruct acts as Generator, Reflector, and Solver (ICLR 2026 ACE Paper)
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# ==============================================================================
# Hardware & Memory (Colab T4 GPU, 15GB VRAM)
# ==============================================================================
QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
}
SOLVER_MAX_NEW_TOKENS = 512

# Image resolution bounds to cap visual tokens and prevent attention OOM on T4
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

# ==============================================================================
# Dataset & Paths
# ==============================================================================
DATASET_NAME = "umd-zhou-lab/ColorBench"
DEFAULT_OUTPUT_DIR = "./results"

# Default number of iterative prompting rounds
DEFAULT_ITERATIONS = 2
