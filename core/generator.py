"""Generator agent for Agentic Context Engineering (ACE)."""

import json
from typing import Dict, List, Any, Optional
from PIL import Image

from .base import BaseAgent
from .playbook import Playbook
from prompts.generator import GENERATOR_SYSTEM_PROMPT
from config import GENERATOR_MAX_NEW_TOKENS


class Generator(BaseAgent):
    """Generator: Explores questions and forms reasoning trajectories using the ACE playbook."""

    def generate_trajectory(
        self,
        question: str,
        choices: List[str],
        image: Optional[Image.Image] = None,
        playbook: Optional[Playbook] = None,
    ) -> Dict[str, Any]:
        """Produce a reasoning trajectory and candidate answer using current playbook context."""
        labels = ["A", "B", "C", "D", "E", "F"]
        options_text = "\n".join(f"({labels[i]}) {c}" for i, c in enumerate(choices))

        user_prompt_parts = []
        if playbook and not playbook.is_empty():
            user_prompt_parts.append(playbook.format_for_prompt())
            user_prompt_parts.append("\n" + "=" * 40 + "\n")

        user_prompt_parts.append(
            f"Question: {question}\nChoices:\n{options_text}\n\n"
            f"Analyze the image, reference any relevant playbook bullets, and provide your reasoning trajectory in JSON."
        )
        user_prompt = "\n".join(user_prompt_parts)

        raw_result = self._call_model(
            system_prompt=GENERATOR_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            image=image,
            max_new_tokens=GENERATOR_MAX_NEW_TOKENS,
        )

        # Normalize proposed_choice through the same option parser as the Solver
        # to prevent junk like "(B) 1" or "(C) No" from being used as predictions.
        raw_choice = raw_result.get("proposed_choice", "")
        from qwen_solver import QwenSolver
        proposed_choice = QwenSolver._parse_option_letter(str(raw_choice), choices=choices)

        return {
            "used_bullet_ids": raw_result.get("used_bullet_ids", []),
            "visual_observations": raw_result.get("visual_observations", ""),
            "reasoning_trajectory": raw_result.get("reasoning_trajectory", ""),
            "proposed_choice": proposed_choice,
        }
