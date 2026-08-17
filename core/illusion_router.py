"""Question-type router and surrogate verifier for Color Illusion.

CoEvoSkill insight applied to VQA:
  - Different illusion subtypes need different strategies. A single flat
    playbook conflates them, producing contradictory bullets.
  - Before committing a new bullet, probe it against a small held-aside
    verification set from the same subtype. Only add if it helps.

Illusion subtypes
-----------------
  UNIFORMITY   "Does the horizontal bar have a uniform color?"
  COMPARISON   "Does color A and B have the same color?" / same-shade / etc.
  RANKING      "Which circles has the darkest color?"
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image


# ---------------------------------------------------------------------------
# Subtype detection
# ---------------------------------------------------------------------------

SUBTYPE_UNIFORMITY = "uniformity"   # horizontal bar / uniform color questions
SUBTYPE_COMPARISON = "comparison"   # same/different color between two patches
SUBTYPE_RANKING    = "ranking"      # which is darkest/lightest/most-different

_UNIFORMITY_PATTERNS = [
    r"uniform",
    r"horizontal bar",
    r"same color in all",
    r"diagonal lines.*same color",
]
_COMPARISON_PATTERNS = [
    r"color a.*color b",
    r"same color\?",
    r"same shade",
    r"different colors?\?",
    r"both eyes",
    r"two pills",
    r"blocks labeled",
    r"two figures.*same",
    r"figures.*different",
]
_RANKING_PATTERNS = [
    r"darkest",
    r"lightest",
    r"brightest",
    r"which.*has the (most|least)",
]


def classify_illusion_question(question: str) -> str:
    """Return one of SUBTYPE_UNIFORMITY, SUBTYPE_COMPARISON, SUBTYPE_RANKING."""
    q = question.lower()
    for pat in _UNIFORMITY_PATTERNS:
        if re.search(pat, q):
            return SUBTYPE_UNIFORMITY
    for pat in _RANKING_PATTERNS:
        if re.search(pat, q):
            return SUBTYPE_RANKING
    # Default: most illusion questions are comparisons
    return SUBTYPE_COMPARISON


# ---------------------------------------------------------------------------
# Surrogate verifier probe
# ---------------------------------------------------------------------------

class SurrogateVerifier:
    """
    Lightweight CoEvoSkill-inspired surrogate verifier for VQA.

    Before a new bullet candidate is committed to the playbook, the verifier
    re-runs the Solver on a small probe set (from the same question subtype)
    WITH vs WITHOUT the candidate, and returns whether accuracy improved.

    This replaces the blind Jaccard-dedup + net-utility counter with an
    empirical accuracy signal on unseen examples.

    Design constraints for T4 GPU
    ------------------------------
    - Probe set is capped at PROBE_SIZE examples (default 3) to keep runtime
      under ~30 extra seconds per adaptation step.
    - The verifier is stateless — it never modifies the playbook directly.
    - It only evaluates ONE candidate at a time to keep memory usage flat.
    """

    PROBE_SIZE: int = 3

    def __init__(self, solver: Any):
        self.solver = solver

    def should_add_bullet(
        self,
        candidate_content: str,
        candidate_category: str,
        current_playbook_text: str,
        probe_items: List[Dict[str, Any]],
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Return (should_add, diagnostic) where should_add is True if the
        candidate bullet improves accuracy on the probe set vs. not having it.

        Parameters
        ----------
        candidate_content    : the proposed bullet text
        candidate_category   : category label
        current_playbook_text: formatted playbook WITHOUT the candidate
        probe_items          : list of dicts with keys image, question, choices, answer
        """
        if not probe_items:
            # No probe data — fall through to normal ACE ADD logic
            return True, {"reason": "no_probe_data", "delta": 0}

        items = probe_items[:self.PROBE_SIZE]

        # Score WITHOUT candidate
        score_without = self._score(current_playbook_text, items)

        # Build playbook WITH candidate appended
        candidate_line = f"\n- [CANDIDATE] {candidate_content}"
        playbook_with = current_playbook_text + candidate_line if current_playbook_text else (
            f"[ACE Context Playbook - Accumulated Domain Strategies]\n\n"
            f"# Category: {candidate_category.replace('_', ' ').title()}\n"
            f"- [CANDIDATE] {candidate_content}"
        )
        score_with = self._score(playbook_with, items)

        delta = score_with - score_without
        should_add = delta >= 0  # add if neutral or better; reject only if strictly harmful

        return should_add, {
            "score_without": score_without,
            "score_with": score_with,
            "delta": delta,
            "probe_count": len(items),
            "reason": "probe_improved" if delta > 0 else ("probe_neutral" if delta == 0 else "probe_hurt"),
        }

    def _score(self, playbook_text: str, items: List[Dict[str, Any]]) -> float:
        """Run Solver on items with given playbook and return accuracy 0–1."""
        if not items:
            return 0.0
        correct = 0
        for item in items:
            skill = playbook_text if playbook_text else None
            mode = "ace" if skill else "baseline"
            res = self.solver.solve(
                image=item["image"],
                question=item["question"],
                choices=item["choices"],
                mode=mode,
                skill=skill,
            )
            if res["prediction"] == item["answer"]:
                correct += 1
        return correct / len(items)
