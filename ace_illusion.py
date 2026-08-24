"""ACE-Illusion: CoEvoSkill-inspired orchestrator for Color Illusion.

Key differences from the base ACE class
-----------------------------------------
1. Per-subtype playbooks (uniformity / comparison / ranking).
   Each question type evolves its own playbook. Bullets for horizontal-bar
   uniformity never contaminate circle-ranking bullets.

2. Surrogate Verifier probe before committing new bullets.
   Inspired by CoEvoSkills' Surrogate Verifier: before adding a new delta
   candidate to a subtype playbook, the verifier re-runs the Solver on a
   small probe set (3 examples from the same subtype held aside during
   adaptation) WITH and WITHOUT the candidate. The candidate is only added
   if it does not hurt accuracy (delta >= 0).

3. Everything else (Generator, Reflector, Curator, ground-truth signal) is
   unchanged — this is an additive improvement on top of the fixed ACE base.
"""

import os
from typing import Dict, List, Any, Optional
from PIL import Image

from core import Generator, Reflector, Curator, Playbook
from core.illusion_router import classify_illusion_question, SurrogateVerifier
from core.subtype_playbook import SubtypePlaybookManager


class ACEIllusion:
    """
    ACE orchestrator specialised for Color Illusion tasks.

    Uses per-subtype playbooks and a surrogate verifier to overcome the
    cross-contamination and noise problems of the flat ACE playbook on
    diverse illusion question types.
    """

    # Fraction of adaptation items held aside per subtype as surrogate probe set.
    # E.g. 0.2 means the first 20% of each subtype's examples are probe-only
    # (not used for playbook updates, only for verifier scoring).
    PROBE_FRACTION: float = 0.20

    def __init__(
        self,
        solver: Any = None,
        task: str = "Color Illusion",
        playbook_base_path: Optional[str] = None,
    ):
        self.solver = solver
        self.task = task
        self.generator = Generator(solver=solver)
        self.reflector = Reflector(solver=solver)
        self.curator = Curator(solver=solver)
        self.verifier = SurrogateVerifier(solver=solver)

        if playbook_base_path:
            self.manager = SubtypePlaybookManager.load_all(playbook_base_path, task=task)
        else:
            self.manager = SubtypePlaybookManager(task=task)
            self._seed_comparison_playbook()
            print(f"[ACEIllusion] Initialized 3 subtype playbooks for task: {task}")

        # Per-subtype probe sets (populated by set_probe_items before adaptation)
        self._probe_items: Dict[str, List[Dict[str, Any]]] = {
            "uniformity": [], "comparison": [], "ranking": []
        }

    def set_probe_items(self, items: List[Dict[str, Any]]):
        """
        Partition adaptation items into per-subtype probe sets.

        The first PROBE_FRACTION of each subtype's examples are reserved as
        the surrogate verifier's probe set. They are NOT used for playbook
        updates — only for verifying candidate bullets.

        Call this ONCE before starting the adaptation loop, passing all
        adaptation items. Returns the remaining items to adapt on.
        """
        from collections import defaultdict
        by_subtype = defaultdict(list)
        for item in items:
            st = classify_illusion_question(item["question"])
            by_subtype[st].append(item)

        adapt_items = []
        for st, st_items in by_subtype.items():
            n_probe = max(1, int(len(st_items) * self.PROBE_FRACTION))
            self._probe_items[st] = st_items[:n_probe]
            adapt_items.extend(st_items[n_probe:])

        # Restore original order by index so logs are consistent
        adapt_items.sort(key=lambda x: x["idx"])

        total_probe = sum(len(v) for v in self._probe_items.values())
        print(f"[ACEIllusion] Probe sets: {dict((k, len(v)) for k, v in self._probe_items.items())} "
              f"({total_probe} total). Adapting on {len(adapt_items)} items.")
        return adapt_items

    def adapt_on_example(
        self,
        question: str,
        choices: List[str],
        image: Optional[Image.Image] = None,
        step_index: Optional[int] = None,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        One ACE adaptation cycle with subtype routing and surrogate verification.

        1. Route question to its subtype playbook.
        2. Generator produces trajectory using that subtype's playbook.
        3. Guard against empty trajectory.
        4. Reflector critiques with GT signal.
        5. Surrogate Verifier probes each delta candidate before Curator adds it.
        6. Curator applies only verified candidates.
        """
        subtype = classify_illusion_question(question)
        playbook = self.manager.get_playbook_for(question)

        # Step 1: Generator
        trajectory = self.generator.generate_trajectory(
            question=question,
            choices=choices,
            image=image,
            playbook=playbook,
        )
        adapted_pred = trajectory.get("proposed_choice", "") or "(UNKNOWN)"

        # Guard: empty trajectory
        if not trajectory.get("reasoning_trajectory") and not trajectory.get("visual_observations"):
            print(f"[ACEIllusion] Warning: Empty Generator trajectory at step {step_index} (subtype={subtype}). Skipping.")
            return self._empty_record(trajectory, adapted_pred, playbook, subtype)

        # Step 2: Reflector
        reflection = self.reflector.reflect(
            question=question,
            choices=choices,
            trajectory=trajectory,
            solver_prediction=adapted_pred,
            solver_raw_output=trajectory.get("reasoning_trajectory", ""),
            image=image,
            playbook=playbook,
            ground_truth=ground_truth,
        )

        # Step 3: Surrogate Verifier — filter delta candidates before curation
        probe_items = self._probe_items.get(subtype, [])
        verified_candidates = []
        verifier_reports = []

        for cand in reflection.get("delta_candidates", []):
            content = cand.get("content", "").strip()
            category = cand.get("category", "general_strategy")
            if not content:
                continue

            should_add, diag = self.verifier.should_commit_candidate(
                candidate=cand,
                playbook=playbook,
                reflection=reflection,
                probe_items=probe_items,
                step_index=step_index,
            )
            verifier_reports.append({"candidate": content, "verdict": should_add, "diag": diag})

            if should_add:
                verified_candidates.append(cand)
            else:
                print(f"[ACEIllusion] Rejected candidate (subtype={subtype}, "
                      f"reason={diag['reason']}): {content[:60]}...")

        # Replace candidates with only verified ones before Curator sees them
        filtered_reflection = dict(reflection)
        filtered_reflection["delta_candidates"] = verified_candidates

        # Step 4: Curator
        curation_report = self.curator.curate(
            playbook=playbook,
            reflection=filtered_reflection,
            step_index=step_index,
        )

        return {
            "subtype": subtype,
            "trajectory": trajectory,
            "solver_prediction": adapted_pred,
            "solver_raw_output": trajectory.get("reasoning_trajectory", ""),
            "reflection": reflection,
            "verifier_reports": verifier_reports,
            "curation_report": curation_report,
            "playbook_version": playbook.version,
            "total_bullets": len(playbook.bullets),
            "subtype_summary": self.manager.summary(),
        }

    def solve_held_out(
        self,
        question: str,
        choices: List[str],
        image: Optional[Image.Image] = None,
    ) -> Dict[str, Any]:
        """Solve a held-out instance using the correct subtype's frozen playbook."""
        subtype = classify_illusion_question(question)
        skill_context = self.manager.format_for_prompt(question)
        res = self.solver.solve(
            image=image,
            question=question,
            choices=choices,
            mode="ace" if skill_context else "baseline",
            skill=skill_context,
        )
        res["subtype"] = subtype
        return res

    def save_playbooks(self, base_json_path: str):
        self.manager.save_all(base_json_path)
        print(f"[ACEIllusion] Saved subtype playbooks to {base_json_path} (3 files).")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _empty_record(self, trajectory, adapted_pred, playbook, subtype):
        return {
            "subtype": subtype,
            "trajectory": trajectory,
            "solver_prediction": adapted_pred,
            "solver_raw_output": "",
            "reflection": {
                "critique": "Skipped — empty trajectory",
                "helpful_bullet_ids": [], "harmful_bullet_ids": [], "delta_candidates": []
            },
            "verifier_reports": [],
            "curation_report": {
                "helpful_ids_updated": [], "harmful_ids_updated": [],
                "added_bullet_ids": [], "updated_bullet_ids": [],
                "reinforced_bullet_ids": [], "suppressed_bullet_ids": [],
                "operations": [],
                "active_bullets_count": sum(
                    1 for b in playbook.bullets.values() if not b.is_suppressed()
                ),
                "total_bullets_count": len(playbook.bullets),
                "playbook_version": playbook.version,
                "end_of_step_summary": [],
            },
            "playbook_version": playbook.version,
            "total_bullets": len(playbook.bullets),
            "subtype_summary": self.manager.summary(),
        }

    def _seed_comparison_playbook(self):
        from core.illusion_router import SUBTYPE_COMPARISON
        seed_bullets = [
            (
                "Compare target-patch mean luminance after isolating each patch from the surrounding background.",
                "procedural_inspection",
            ),
            (
                "Account for illumination gradients and shadow boundaries that can shift perceived hue.",
                "confounder_handling",
            ),
            (
                "Check whether a global color cast affects both target regions equally.",
                "confounder_handling",
            ),
        ]
        pb = self.manager.playbooks[SUBTYPE_COMPARISON]
        for content, category in seed_bullets:
            pb.add_bullet(category=category, content=content, source_step=None)
