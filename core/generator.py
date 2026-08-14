import json
from typing import Dict, List, Any, Optional
from PIL import Image

from .base import BaseAgent
from prompts.generator import GENERATOR_PROMPT

class Generator(BaseAgent):
    """Pass 1 (Generate): Create initial cognitive skill for the question using local Qwen."""

    def generate_skill(
        self,
        question: str,
        choices: List[str],
        image: Optional[Image.Image] = None,
    ) -> Dict[str, Any]:
        """Generate an initial skill to solve the VQA question."""
        prompt = f"Question: {question}\nOptions: {json.dumps(choices)}"
        return self._call_model(GENERATOR_PROMPT, prompt, image=image)
