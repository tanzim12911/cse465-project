"""Curator agent for Agentic Context Engineering (ACE).

The Curator consolidates lessons from the Reflector and applies deterministic
grow-and-refine operations (ADD, UPDATE, and suppression) to the persistent Playbook.
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
        2. Refine (UPDATE): update existing bullets in-place if refines_bullet_id is specified.
        3. Grow (ADD): append new non-duplicate delta candidates.
        4. Track conservative suppression of net-negative bullets.
        """
        helpful_ids = reflection.get("helpful_bullet_ids", [])
        harmful_ids = reflection.get("harmful_bullet_ids", [])
        delta_candidates = reflection.get("delta_candidates", [])

        # 1. Update utility counters
        playbook.mark_helpful(helpful_ids)
        playbook.mark_harmful(harmful_ids)

        operations = []
        added_bullet_ids = []
        updated_bullet_ids = []
        reinforced_bullet_ids = []

        # 2. Integrate delta candidates
        for cand in delta_candidates:
            cat = cand.get("category", "general_strategy")
            content = cand.get("content", "").strip()
            refines_id = cand.get("refines_bullet_id")
            if not content:
                continue

            # Case A: In-Place Refinement (UPDATE)
            if refines_id and refines_id in playbook.bullets:
                update_info = playbook.update_bullet(
                    bullet_id=refines_id,
                    new_content=content,
                    new_category=cat,
                    source_step=step_index,
                )
                if update_info:
                    updated_bullet_ids.append(refines_id)
                    operations.append({
                        "op_type": "UPDATE",
                        "bullet_id": refines_id,
                        "prev_content": update_info["prev_content"],
                        "new_content": update_info["new_content"],
                        "helpful_count": update_info["helpful_count"],
                        "harmful_count": update_info["harmful_count"],
                        "refinement_count": update_info["refinement_count"],
                        "is_suppressed": playbook.bullets[refines_id].is_suppressed(),
                    })
                continue

            # Case B: Addition or Reinforcement (ADD)
            res_id = playbook.add_bullet(
                category=cat,
                content=content,
                source_step=step_index,
                dedup_threshold=0.65,
            )
            if res_id:
                bullet = playbook.bullets[res_id]
                if bullet.source_step == step_index and bullet.refinement_count == 0 and bullet.helpful_count == 0:
                    added_bullet_ids.append(res_id)
                    operations.append({
                        "op_type": "ADD",
                        "bullet_id": res_id,
                        "content": content,
                        "helpful_count": bullet.helpful_count,
                        "harmful_count": bullet.harmful_count,
                        "is_suppressed": bullet.is_suppressed(),
                    })
                else:
                    reinforced_bullet_ids.append(res_id)
                    operations.append({
                        "op_type": "REINFORCE",
                        "bullet_id": res_id,
                        "content": bullet.content,
                        "helpful_count": bullet.helpful_count,
                        "harmful_count": bullet.harmful_count,
                        "is_suppressed": bullet.is_suppressed(),
                    })

        suppressed_ids = playbook.get_suppressed_bullet_ids()

        return {
            "helpful_ids_updated": helpful_ids,
            "harmful_ids_updated": harmful_ids,
            "added_bullet_ids": added_bullet_ids,
            "updated_bullet_ids": updated_bullet_ids,
            "reinforced_bullet_ids": reinforced_bullet_ids,
            "suppressed_bullet_ids": suppressed_ids,
            "operations": operations,
            "active_bullets_count": len(playbook.bullets) - len(suppressed_ids),
            "total_bullets_count": len(playbook.bullets),
            "playbook_version": playbook.version,
        }
