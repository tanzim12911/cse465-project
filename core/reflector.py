import json
from typing import Dict, List, Any, Optional
from PIL import Image

from .base import BaseAgent
from prompts.reflector import REFLECTOR_PROMPT

class Reflector(BaseAgent):
    """Pass 2+ (Reflect & Refine): Evaluate previous answer and improve skill using local Qwen."""

    def reflect_and_refine(
        self,
        question: str,
        choices: List[str],
        prev_skill: str,
        prev_answer: str,
        prev_raw_output: str,
        image: Optional[Image.Image] = None,
    ) -> Dict[str, Any]:
        """Reflect on the skill's effectiveness and generate a refined skill."""
        prompt = (
            f"Original Question: {question}\n"
            f"Options: {json.dumps(choices)}\n\n"
            f"Previous Skill Generated: {prev_skill}\n\n"
            f"VLM's Answer: {prev_answer}\n"
            f"VLM's Raw Output: {prev_raw_output[:300]}\n\n"
            f"Reflect on the skill's effectiveness and generate a refined, more targeted skill."
        )
        return self._call_model(REFLECTOR_PROMPT, prompt, image=image)
