"""Reflector agent for Agentic Context Engineering (ACE)."""

import json
from typing import Dict, List, Any, Optional
from PIL import Image

from .base import BaseAgent
from .playbook import Playbook
from prompts.reflector import REFLECTOR_SYSTEM_PROMPT


class Reflector(BaseAgent):
    """Reflector: Evaluates execution traces and distills reusable delta lessons."""

    def reflect(
        self,
        question: str,
        choices: List[str],
        trajectory: Dict[str, Any],
        solver_prediction: str,
        solver_raw_output: str,
        image: Optional[Image.Image] = None,
        playbook: Optional[Playbook] = None,
    ) -> Dict[str, Any]:
        """Critique the attempt and extract candidate delta updates."""
        labels = ["A", "B", "C", "D", "E", "F"]
        options_text = "\n".join(f"({labels[i]}) {c}" for i, c in enumerate(choices))

        user_prompt_parts = []
        if playbook and not playbook.is_empty():
            user_prompt_parts.append(playbook.format_for_prompt())
            user_prompt_parts.append("\n" + "=" * 40 + "\n")

        user_prompt_parts.append(
            f"Question: {question}\n"
            f"Choices:\n{options_text}\n\n"
            f"Generator's Visual Observations:\n{trajectory.get('visual_observations', 'None')}\n\n"
            f"Generator's Trajectory:\n{trajectory.get('reasoning_trajectory', 'None')}\n\n"
            f"Generator's Proposed Choice: {trajectory.get('proposed_choice', 'None')}\n\n"
            f"Solver's Output: {solver_prediction}\n"
            f"Solver's Trace: {solver_raw_output[:300]}\n\n"
            f"Critique this attempt. Identify which existing bullets were helpful/harmful, and propose 1-2 concise delta candidate rules."
        )
        user_prompt = "\n".join(user_prompt_parts)

        raw_result = self._call_model(
            system_prompt=REFLECTOR_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            image=image,
            max_new_tokens=256,
        )

        # Normalize delta_candidates
        delta_cands = raw_result.get("delta_candidates", [])
        if isinstance(delta_cands, dict):
            delta_cands = [delta_cands]
        elif not isinstance(delta_cands, list):
            delta_cands = []

        clean_cands = []
        for cand in delta_cands:
            if isinstance(cand, dict) and "content" in cand:
                ref_id = cand.get("refines_bullet_id")
                if ref_id and isinstance(ref_id, str):
                    ref_id = ref_id.strip()
                else:
                    ref_id = None

                clean_cands.append({
                    "category": cand.get("category", "general_strategy").strip().lower(),
                    "content": cand.get("content", "").strip(),
                    "refines_bullet_id": ref_id,
                })

        return {
            "critique": raw_result.get("critique", ""),
            "helpful_bullet_ids": raw_result.get("helpful_bullet_ids", []),
            "harmful_bullet_ids": raw_result.get("harmful_bullet_ids", []),
            "delta_candidates": clean_cands,
        }
