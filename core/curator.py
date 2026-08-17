"""Curator agent for Agentic Context Engineering (ACE).

The Curator consolidates lessons from the Reflector and applies deterministic
grow-and-refine operations (ADD, UPDATE, and suppression) to the persistent Playbook.

Conservative playbook maintenance rules
---------------------------------------
UPDATE gating
  A candidate with refines_bullet_id is applied as UPDATE only when:
    - the target bullet exists, AND
    - target.harmful_count >= 2  (at least two harmful attributions), AND
    - the target bullet is NOT high-utility.
  Otherwise the UPDATE is rejected and the candidate is re-evaluated for ADD
  (or discarded as a duplicate if it substantially overlaps the target).

ADD capacity cap
  Maximum active bullets = MAX_ACTIVE_BULLETS (15).
  When at capacity a new ADD is logged as ADD_REJECTED_CAPACITY.
  Existing bullets are NEVER deleted to make room.

Suppression
  Delegated to PlaybookBullet.is_suppressed() which now requires:
    - harmful_count - helpful_count >= 2, AND
    - harmful_count >= 2, AND
    - bullet is NOT high-utility.
  Suppression deactivates the bullet from Solver context; it is never deleted.

Counter preservation
  update_bullet() in playbook.py already preserves helpful_count and harmful_count.
  refinement_count is incremented there on every accepted UPDATE.

Operation log event types
  ADD                                – new bullet added
  ADD_REJECTED_DUPLICATE             – candidate too similar to an existing bullet
  ADD_REJECTED_CAPACITY              – active bullet cap reached
  UPDATE_ACCEPTED                    – in-place refinement applied
  UPDATE_REJECTED_HIGH_UTILITY       – target is high-utility; UPDATE blocked
  UPDATE_REJECTED_INSUFFICIENT_EVIDENCE – target.harmful_count < 2
  KEPT                               – candidate discarded (overlaps target, no new info)
  SUPPRESSED                         – existing bullet entered suppressed state this step
  REINFORCE                          – duplicate candidate reinforced an existing bullet
"""

from typing import Dict, List, Any, Optional
from .base import BaseAgent
from .playbook import Playbook

MAX_ACTIVE_BULLETS: int = 20
# Jaccard similarity threshold used in update-rejected ADD-fallback overlap check.
# Intentionally the same threshold as add_bullet() so behaviour is symmetric.
_OVERLAP_THRESHOLD: float = 0.65


class Curator(BaseAgent):
    """Curator: Deterministically merges delta updates and maintains playbook health."""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def curate(
        self,
        playbook: Playbook,
        reflection: Dict[str, Any],
        step_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Apply reflection insights to the playbook.

        Steps
        -----
        1. Credit assignment  – mark helpful & harmful bullets.
        2. Delta integration  – process each candidate (UPDATE or ADD with gating).
        3. Suppression check  – identify bullets that entered suppressed state.
        4. Summary            – return structured curation report.
        """
        helpful_ids: List[str] = reflection.get("helpful_bullet_ids", [])
        harmful_ids: List[str] = reflection.get("harmful_bullet_ids", [])
        delta_candidates: List[Dict[str, Any]] = reflection.get("delta_candidates", [])

        # ---- Step 1: Update utility counters ----
        playbook.mark_helpful(helpful_ids)
        playbook.mark_harmful(harmful_ids)

        operations: List[Dict[str, Any]] = []
        added_bullet_ids: List[str] = []
        updated_bullet_ids: List[str] = []
        reinforced_bullet_ids: List[str] = []

        # ---- Step 2: Integrate delta candidates ----
        for cand in delta_candidates:
            cat = cand.get("category", "general_strategy")
            content = cand.get("content", "").strip()
            refines_id = cand.get("refines_bullet_id")

            if not content:
                continue

            if refines_id and refines_id in playbook.bullets:
                # ---- Case A: Attempted UPDATE ----
                op = self._handle_update(
                    playbook=playbook,
                    refines_id=refines_id,
                    cat=cat,
                    content=content,
                    step_index=step_index,
                )
                operations.append(op)
                if op["op_type"] == "UPDATE_ACCEPTED":
                    updated_bullet_ids.append(refines_id)
                # For UPDATE_REJECTED_* cases the candidate is re-evaluated
                # for ADD (or KEPT/discarded) inside _handle_update_fallback.
                elif op["op_type"] in (
                    "UPDATE_REJECTED_HIGH_UTILITY",
                    "UPDATE_REJECTED_INSUFFICIENT_EVIDENCE",
                ):
                    fallback_op = self._handle_update_fallback(
                        playbook=playbook,
                        refines_id=refines_id,
                        cat=cat,
                        content=content,
                        step_index=step_index,
                    )
                    if fallback_op:
                        operations.append(fallback_op)
                        if fallback_op["op_type"] == "ADD":
                            added_bullet_ids.append(fallback_op["bullet_id"])
                        elif fallback_op["op_type"] == "REINFORCE":
                            reinforced_bullet_ids.append(fallback_op["bullet_id"])
            else:
                # ---- Case B: Fresh ADD (no refines_bullet_id) ----
                add_op = self._handle_add(
                    playbook=playbook,
                    cat=cat,
                    content=content,
                    step_index=step_index,
                )
                operations.append(add_op)
                if add_op["op_type"] == "ADD":
                    added_bullet_ids.append(add_op["bullet_id"])
                elif add_op["op_type"] == "REINFORCE":
                    reinforced_bullet_ids.append(add_op["bullet_id"])

        # ---- Step 3: Suppression audit ----
        suppressed_ids = self._audit_suppression(playbook, operations, step_index)

        # ---- Step 4: End-of-step playbook summary ----
        end_of_step_summary = self._build_summary(playbook)

        active_count = sum(
            1 for b in playbook.bullets.values() if not b.is_suppressed()
        )

        return {
            "helpful_ids_updated": helpful_ids,
            "harmful_ids_updated": harmful_ids,
            "added_bullet_ids": added_bullet_ids,
            "updated_bullet_ids": updated_bullet_ids,
            "reinforced_bullet_ids": reinforced_bullet_ids,
            "suppressed_bullet_ids": suppressed_ids,
            "operations": operations,
            "active_bullets_count": active_count,
            "total_bullets_count": len(playbook.bullets),
            "playbook_version": playbook.version,
            "end_of_step_summary": end_of_step_summary,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _handle_update(
        self,
        playbook: Playbook,
        refines_id: str,
        cat: str,
        content: str,
        step_index: Optional[int],
    ) -> Dict[str, Any]:
        """Evaluate and optionally apply an UPDATE to an existing bullet.

        Returns a single operation dict with op_type set to one of:
          UPDATE_ACCEPTED
          UPDATE_REJECTED_HIGH_UTILITY
          UPDATE_REJECTED_INSUFFICIENT_EVIDENCE
        """
        target = playbook.bullets[refines_id]

        # Guard 1: high-utility bullet — UPDATE is always rejected
        if target.is_high_utility():
            return {
                "op_type": "UPDATE_REJECTED_HIGH_UTILITY",
                "bullet_id": refines_id,
                "old_content": target.content,
                "candidate_content": content,
                "helpful_count": target.helpful_count,
                "harmful_count": target.harmful_count,
                "refinement_count": target.refinement_count,
                "reason": (
                    f"Target {refines_id} is high-utility "
                    f"(helpful={target.helpful_count}, harmful={target.harmful_count}). "
                    "UPDATE rejected to preserve strong positive history."
                ),
            }

        # Guard 2: insufficient evidence — require at least 2 harmful attributions
        if target.harmful_count < 2:
            return {
                "op_type": "UPDATE_REJECTED_INSUFFICIENT_EVIDENCE",
                "bullet_id": refines_id,
                "old_content": target.content,
                "candidate_content": content,
                "helpful_count": target.helpful_count,
                "harmful_count": target.harmful_count,
                "refinement_count": target.refinement_count,
                "reason": (
                    f"Target {refines_id} has only {target.harmful_count} harmful "
                    "attribution(s); minimum 2 required for UPDATE."
                ),
            }

        # UPDATE approved — counters are preserved by update_bullet()
        update_info = playbook.update_bullet(
            bullet_id=refines_id,
            new_content=content,
            new_category=cat,
            source_step=step_index,
        )
        if update_info is None:
            # update_bullet returned None (empty/short content) — treat as rejected
            return {
                "op_type": "UPDATE_REJECTED_INSUFFICIENT_EVIDENCE",
                "bullet_id": refines_id,
                "old_content": target.content,
                "candidate_content": content,
                "helpful_count": target.helpful_count,
                "harmful_count": target.harmful_count,
                "refinement_count": target.refinement_count,
                "reason": "Candidate content too short or empty; UPDATE aborted.",
            }

        return {
            "op_type": "UPDATE_ACCEPTED",
            "bullet_id": refines_id,
            "old_content": update_info["prev_content"],
            "new_content": update_info["new_content"],
            "helpful_count": update_info["helpful_count"],
            "harmful_count": update_info["harmful_count"],
            "refinement_count": update_info["refinement_count"],
            "is_suppressed": playbook.bullets[refines_id].is_suppressed(),
        }

    def _handle_update_fallback(
        self,
        playbook: Playbook,
        refines_id: str,
        cat: str,
        content: str,
        step_index: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        """After a rejected UPDATE, decide whether to ADD or KEPT.

        Algorithm
        ---------
        1. Compute overlap between candidate content and the target bullet.
        2. If overlap >= _OVERLAP_THRESHOLD → KEPT (candidate adds no new info).
        3. Otherwise → attempt ADD (subject to capacity cap).
        """
        target = playbook.bullets[refines_id]
        similarity = Playbook._compute_similarity(content, target.content)

        if similarity >= _OVERLAP_THRESHOLD:
            # Candidate substantially overlaps the original bullet — discard it.
            return {
                "op_type": "KEPT",
                "bullet_id": refines_id,
                "candidate_content": content,
                "target_content": target.content,
                "similarity": round(similarity, 3),
                "reason": (
                    f"Candidate overlaps {refines_id} (similarity={similarity:.2f}). "
                    "Discarded to prevent contradictory twin bullets."
                ),
            }

        # Candidate carries genuinely new information → try to ADD
        return self._handle_add(
            playbook=playbook,
            cat=cat,
            content=content,
            step_index=step_index,
        )

    def _handle_add(
        self,
        playbook: Playbook,
        cat: str,
        content: str,
        step_index: Optional[int],
    ) -> Dict[str, Any]:
        """Attempt to add a new bullet, respecting the active-bullet capacity cap.

        Returns one of: ADD, ADD_REJECTED_DUPLICATE, ADD_REJECTED_CAPACITY, REINFORCE.
        """
        # Count currently active (non-suppressed) bullets
        active_count = sum(
            1 for b in playbook.bullets.values() if not b.is_suppressed()
        )
        if active_count >= MAX_ACTIVE_BULLETS:
            return {
                "op_type": "ADD_REJECTED_CAPACITY",
                "candidate_content": content,
                "active_count": active_count,
                "capacity": MAX_ACTIVE_BULLETS,
                "reason": (
                    f"Active playbook at capacity ({active_count}/{MAX_ACTIVE_BULLETS}). "
                    "No bullets deleted; new ADD skipped."
                ),
            }

        prev_total = len(playbook.bullets)
        pre_counts = {
            bid: b.helpful_count for bid, b in playbook.bullets.items()
        }

        res_id = playbook.add_bullet(
            category=cat,
            content=content,
            source_step=step_index,
            dedup_threshold=_OVERLAP_THRESHOLD,
            reinforce_on_dedup=False,  # credit is managed via mark_helpful/mark_harmful
        )

        if res_id is None:
            # add_bullet returned None — content too short, shouldn't normally happen
            return {
                "op_type": "ADD_REJECTED_DUPLICATE",
                "candidate_content": content,
                "reason": "Content rejected by add_bullet (too short or None returned).",
            }

        bullet = playbook.bullets[res_id]

        # Determine if this was a reinforcement (existing bullet's helpful_count bumped)
        # or a true new addition.
        if len(playbook.bullets) == prev_total and pre_counts.get(res_id, -1) < bullet.helpful_count:
            # add_bullet hit the dedup path and incremented helpful_count of an existing bullet
            return {
                "op_type": "REINFORCE",
                "bullet_id": res_id,
                "content": bullet.content,
                "helpful_count": bullet.helpful_count,
                "harmful_count": bullet.harmful_count,
                "is_suppressed": bullet.is_suppressed(),
            }

        if len(playbook.bullets) > prev_total:
            # New bullet was genuinely added
            return {
                "op_type": "ADD",
                "bullet_id": res_id,
                "content": content,
                "helpful_count": bullet.helpful_count,
                "harmful_count": bullet.harmful_count,
                "is_suppressed": bullet.is_suppressed(),
            }

        # Fallback — should not normally reach here
        return {
            "op_type": "ADD_REJECTED_DUPLICATE",
            "candidate_content": content,
            "reason": "Duplicate detected by add_bullet; existing bullet unchanged.",
        }

    def _audit_suppression(
        self,
        playbook: Playbook,
        operations: List[Dict[str, Any]],
        step_index: Optional[int],
    ) -> List[str]:
        """Check for bullets that are now suppressed and append SUPPRESSED ops."""
        suppressed_ids: List[str] = []
        existing_op_bullet_ids = {op.get("bullet_id") for op in operations}

        for bid, bullet in playbook.bullets.items():
            if bullet.is_suppressed():
                suppressed_ids.append(bid)
                # Only append a SUPPRESSED log entry if not already covered by
                # another op this step (avoid double-logging).
                if bid not in existing_op_bullet_ids:
                    operations.append({
                        "op_type": "SUPPRESSED",
                        "bullet_id": bid,
                        "content": bullet.content,
                        "helpful_count": bullet.helpful_count,
                        "harmful_count": bullet.harmful_count,
                        "is_high_utility": bullet.is_high_utility(),
                        "reason": (
                            f"harmful={bullet.harmful_count}, helpful={bullet.helpful_count}, "
                            f"net={bullet.harmful_count - bullet.helpful_count}"
                        ),
                    })

        return suppressed_ids

    @staticmethod
    def _build_summary(playbook: Playbook) -> List[Dict[str, Any]]:
        """Return end-of-step audit record for every persistent bullet."""
        summary = []
        for bullet in playbook.bullets.values():
            summary.append({
                "bullet_id": bullet.bullet_id,
                "content": bullet.content,
                "helpful_count": bullet.helpful_count,
                "harmful_count": bullet.harmful_count,
                "refinement_count": bullet.refinement_count,
                "status": "suppressed" if bullet.is_suppressed() else "active",
                "is_high_utility": bullet.is_high_utility(),
                "source_step": bullet.source_step,
            })
        return summary
