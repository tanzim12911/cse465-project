"""Reflector agent for Agentic Context Engineering (ACE)."""

import json
from typing import Dict, List, Any, Optional
from PIL import Image

from .base import BaseAgent
from .playbook import Playbook
from prompts.reflector import REFLECTOR_SYSTEM_PROMPT
from config import REFLECTOR_MAX_NEW_TOKENS


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
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Critique the attempt and extract candidate delta updates."""
        labels = ["A", "B", "C", "D", "E", "F"]
        options_text = "\n".join(f"({labels[i]}) {c}" for i, c in enumerate(choices))

        # Determine outcome so the Reflector has an objective correctness signal
        if ground_truth:
            is_correct = (solver_prediction.strip().upper() == ground_truth.strip().upper())
            outcome_line = (
                f"Ground Truth: {ground_truth}\n"
                f"Outcome: {'CORRECT ✓' if is_correct else 'INCORRECT ✗'}\n"
            )
        else:
            outcome_line = "Ground Truth: (not provided)\n"

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
            f"{outcome_line}\n"
            f"Critique this attempt. If the answer was INCORRECT, identify which "
            f"bullets misled the reasoning and mark them as harmful. If CORRECT, "
            f"identify which bullets helped and mark them as helpful. "
            f"Propose 1-2 concise delta candidate rules."
        )
        user_prompt = "\n".join(user_prompt_parts)

        raw_result = self._call_model(
            system_prompt=REFLECTOR_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            image=image,
            max_new_tokens=REFLECTOR_MAX_NEW_TOKENS,
        )

        # Normalize delta_candidates
        delta_cands = raw_result.get("delta_candidates", [])
        if isinstance(delta_cands, dict):
            delta_cands = [delta_cands]
        elif not isinstance(delta_cands, list):
            delta_cands = []

        clean_cands = []
        _VALID_RULE_TYPES = {
            "procedural_inspection",
            "conclusion_directed",
            "confounder_handling",
            "task_specific",
        }
        for cand in delta_cands:
            if isinstance(cand, dict) and "content" in cand:
                ref_id = cand.get("refines_bullet_id")
                if ref_id and isinstance(ref_id, str):
                    ref_id = ref_id.strip()
                else:
                    ref_id = None

                raw_rule_type = cand.get("rule_type", "procedural_inspection")
                if isinstance(raw_rule_type, str):
                    raw_rule_type = raw_rule_type.strip().lower()
                rule_type = raw_rule_type if raw_rule_type in _VALID_RULE_TYPES else "procedural_inspection"

                clean_cands.append({
                    "category": cand.get("category", "general_strategy").strip().lower(),
                    "content": cand.get("content", "").strip(),
                    "rule_type": rule_type,
                    "refines_bullet_id": ref_id,
                })


        return {
            "critique": raw_result.get("critique", ""),
            "helpful_bullet_ids": raw_result.get("helpful_bullet_ids", []),
            "harmful_bullet_ids": raw_result.get("harmful_bullet_ids", []),
            "delta_candidates": clean_cands,
        }
