"""Configuration for CSE465 ColorBench Adaptive Skill Generation Pipeline."""

import os

# ==============================================================================
# Model Configurations
# ==============================================================================
# Phase 1: Skill Generator (Gemini Flash via REST or google-genai SDK)
GEMINI_MODEL_ID = "gemini-flash-latest"
GEMINI_FALLBACK_MODEL_ID = "gemini-2.5-flash"

# Phase 2: Vision-Language Solver (Qwen2.5-VL-7B-Instruct, 4-bit quantized)
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# ==============================================================================
# API Key Resolution
# ==============================================================================
def get_gemini_api_key() -> str:
    """Retrieve Gemini API key from Colab userdata or environment variables."""
    try:
        from google.colab import userdata
        key = userdata.get("GEMINI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    raise ValueError(
        "GEMINI_API_KEY not found. Set it via Colab Secrets or "
        "environment variable: export GEMINI_API_KEY='your_key'"
    )

# ==============================================================================
# Hardware & Memory (Colab T4 GPU, 15GB VRAM)
# ==============================================================================
QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
}
SOLVER_MAX_NEW_TOKENS = 256

# Gemini API rate limit delay (seconds between calls)
GEMINI_RPM_DELAY = 4.0

# ==============================================================================
# Dataset & Paths
# ==============================================================================
DATASET_NAME = "umd-zhou-lab/ColorBench"
DEFAULT_OUTPUT_DIR = "./results"

# Default number of iterative prompting rounds
DEFAULT_ITERATIONS = 2
