import json
from typing import Dict, List, Any

from .base import BaseAgent
from prompts.generator import GENERATOR_PROMPT

class Generator(BaseAgent):
    """Pass 1 (Generate): Create initial skill for the question."""
    
    def generate_skill(self, question: str, choices: List[str]) -> Dict[str, Any]:
        """Generate a skill to solve the VQA question."""
        prompt = f"Question: {question}\nOptions: {json.dumps(choices)}"
        return self._call_gemini(GENERATOR_PROMPT, prompt)
