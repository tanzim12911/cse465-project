import json
from typing import Dict, List, Any, Optional
from PIL import Image

from .base import BaseAgent
from prompts.generator import INITIAL_GENERATOR_PROMPT, ITERATIVE_GENERATOR_PROMPT, GENERATOR_PROMPT


class Generator(BaseAgent):
    """Pass 1 and Iterative Skill Synthesizer using local Qwen model."""

    def generate_initial_skill(
        self,
        question: str,
        choices: List[str],
        image: Optional[Image.Image] = None,
    ) -> Dict[str, Any]:
        """Generate an initial cognitive skill to solve the VQA question."""
        prompt = f"Question: {question}\nOptions: {json.dumps(choices)}"
        return self._call_model(INITIAL_GENERATOR_PROMPT, prompt, image=image)

    def refine_skill_with_feedback(
        self,
        question: str,
        choices: List[str],
        prev_skill: str,
        verifier_feedback: str,
        image: Optional[Image.Image] = None,
    ) -> Dict[str, Any]:
        """
        Refine a cognitive skill based strictly on sanitized Verifier diagnostic feedback.
        Boundary Guard: Does NOT receive target ground truth, test cases, or raw solver output.
        """
        prompt = (
            f"Question: {question}\n"
            f"Options: {json.dumps(choices)}\n\n"
            f"Previous Skill:\n{prev_skill}\n\n"
            f"Verifier Diagnostic Feedback:\n{verifier_feedback}\n\n"
            f"Generate a refined, more effective visual reasoning skill."
        )
        return self._call_model(ITERATIVE_GENERATOR_PROMPT, prompt, image=image)

    # Legacy alias for backward compatibility with ace_baseline
    def generate_skill(
        self,
        question: str,
        choices: List[str],
        image: Optional[Image.Image] = None,
    ) -> Dict[str, Any]:
        return self.generate_initial_skill(question, choices, image=image)
