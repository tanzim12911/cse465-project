"""Curator agent for Agentic Context Engineering (ACE).

The Curator consolidates lessons from the Reflector and applies deterministic
grow-and-refine operations to the persistent Playbook.
"""

from typing import Dict, List, Any, Optional
from .base import BaseAgent
from .playbook import Playbook


class Curator(BaseAgent):
    """Curator: Deterministically merges delta updates and maintains playbook health."""

    def curate(
        self,
        playbook: Playbook,
        reflection: Dict[str, Any],
        step_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Apply reflection insights to the playbook:
        1. Credit assignment: mark helpful & harmful bullets.
        2. Grow: append new non-duplicate delta candidates.
        """
        helpful_ids = reflection.get("helpful_bullet_ids", [])
        harmful_ids = reflection.get("harmful_bullet_ids", [])
        delta_candidates = reflection.get("delta_candidates", [])

        # 1. Update utility counters
        playbook.mark_helpful(helpful_ids)
        playbook.mark_harmful(harmful_ids)

        added_bullet_ids = []
        reinforced_bullet_ids = []

        # 2. Integrate delta candidates
        for cand in delta_candidates:
            cat = cand.get("category", "general_strategy")
            content = cand.get("content", "").strip()
            if not content:
                continue

            res_id = playbook.add_bullet(
                category=cat,
                content=content,
                source_step=step_index,
                dedup_threshold=0.70,
            )
            if res_id:
                if res_id in playbook.bullets and playbook.bullets[res_id].source_step == step_index:
                    added_bullet_ids.append(res_id)
                else:
                    reinforced_bullet_ids.append(res_id)

        return {
            "helpful_ids_updated": helpful_ids,
            "harmful_ids_updated": harmful_ids,
            "added_bullet_ids": added_bullet_ids,
            "reinforced_bullet_ids": reinforced_bullet_ids,
            "total_bullets": len(playbook.bullets),
            "playbook_version": playbook.version,
        }
