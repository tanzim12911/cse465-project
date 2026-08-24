"""Routing and strict validation for Color Illusion playbooks."""

import re
from typing import Dict, List, Any, Tuple

from .curator import Curator
from .playbook import Playbook

SUBTYPE_UNIFORMITY = "uniformity"
SUBTYPE_COMPARISON = "comparison"
SUBTYPE_RANKING = "ranking"

_UNIFORMITY_PATTERNS = [r"uniform", r"horizontal bar", r"same color in all", r"diagonal lines.*same color"]
_RANKING_PATTERNS = [r"darkest", r"lightest", r"brightest", r"which.*has the (most|least)"]


def classify_illusion_question(question: str) -> str:
    """Route a Color Illusion question to its non-overlapping playbook."""
    q = question.lower()
    if any(re.search(pattern, q) for pattern in _UNIFORMITY_PATTERNS):
        return SUBTYPE_UNIFORMITY
    if any(re.search(pattern, q) for pattern in _RANKING_PATTERNS):
        return SUBTYPE_RANKING
    return SUBTYPE_COMPARISON


class SurrogateVerifier:
    """Accept only a candidate that improves the committed playbook state.

    The old gate appended text to the prompt and accepted ties on three examples.
    This verifier instead simulates the Curator operation on a clone of the
    playbook, then requires at least one additional correct answer on the fixed
    same-subtype validation probe set.
    """

    MAX_PROBE_SIZE = 8
    MIN_CORRECT_GAIN = 1

    def __init__(self, solver: Any):
        self.solver = solver

    def should_commit_candidate(
        self,
        candidate: Dict[str, Any],
        playbook: Playbook,
        reflection: Dict[str, Any],
        probe_items: List[Dict[str, Any]],
        step_index: int | None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate the actual ADD/UPDATE result, never an appended-text proxy."""
        if not probe_items:
            return False, {"reason": "no_probe_data", "delta": 0, "probe_count": 0}

        proposed = Playbook.from_dict(playbook.to_dict())
        proposal_reflection = {
            "helpful_bullet_ids": reflection.get("helpful_bullet_ids", []),
            "harmful_bullet_ids": reflection.get("harmful_bullet_ids", []),
            "delta_candidates": [candidate],
        }
        report = Curator(solver=None).curate(proposed, proposal_reflection, step_index)
        committed = bool(report["added_bullet_ids"] or report["updated_bullet_ids"] or report["reinforced_bullet_ids"])
        if not committed:
            return False, {
                "reason": "candidate_not_committable",
                "delta": 0,
                "probe_count": 0,
                "operations": report["operations"],
            }

        items = probe_items[:self.MAX_PROBE_SIZE]
        score_without, correct_without = self._score(playbook.format_for_prompt(), items)
        score_with, correct_with = self._score(proposed.format_for_prompt(), items)
        gain = correct_with - correct_without
        accepted = gain >= self.MIN_CORRECT_GAIN
        return accepted, {
            "score_without": score_without,
            "score_with": score_with,
            "correct_without": correct_without,
            "correct_with": correct_with,
            "delta": score_with - score_without,
            "correct_gain": gain,
            "probe_count": len(items),
            "reason": "probe_improved" if accepted else "no_strict_probe_improvement",
            "operations": report["operations"],
        }

    def _score(self, playbook_text: str, items: List[Dict[str, Any]]) -> Tuple[float, int]:
        correct = 0
        for item in items:
            res = self.solver.solve(
                image=item["image"], question=item["question"], choices=item["choices"],
                mode="ace" if playbook_text else "baseline", skill=playbook_text or None,
            )
            correct += int(res["prediction"] == item["answer"])
        return correct / len(items), correct
