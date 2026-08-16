"""ACE Framework Orchestrator for ColorBench.

Implements the faithful ACE architecture (ICLR 2026):
  Generator -> Reflector -> Curator -> Persistent Playbook -> Solver
"""

import os
from typing import Dict, List, Any, Optional
from PIL import Image

from core import Generator, Reflector, Curator, Playbook


class ACE:
    """
    Main ACE System Orchestrator.
    
    Coordinates the Generator, Reflector, and Curator to maintain and evolve
    an itemized context Playbook across dataset instances.
    """

    def __init__(self, solver: Any = None, task: str = "Color Mimicry", playbook_path: Optional[str] = None):
        self.solver = solver
        self.task = task
        self.generator = Generator(solver=solver)
        self.reflector = Reflector(solver=solver)
        self.curator = Curator(solver=solver)

        if playbook_path and os.path.exists(playbook_path):
            self.playbook = Playbook.load(playbook_path)
            print(f"[ACE] Loaded existing playbook from {playbook_path} ({len(self.playbook.bullets)} bullets)")
        else:
            self.playbook = Playbook(task=task)
            print(f"[ACE] Initialized empty playbook for task: {task}")

    def adapt_on_example(
        self,
        question: str,
        choices: List[str],
        image: Optional[Image.Image] = None,
        step_index: Optional[int] = None,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute one full ACE adaptation cycle on a single training/adaptation instance:
        1. Generator explores question with current playbook -> Trajectory + Proposed Answer.
        2. Reflector critiques the Generator's trajectory using the ground truth
           outcome -> Delta Candidates + Bullet Attribution.
        3. Curator applies delta updates -> Evolving Playbook.

        The Generator's proposed_choice is used as the adapted answer so the
        Reflector evaluates the same reasoning it critiques (no separate Solver
        call during adaptation). The Solver is reserved for held-out evaluation.
        """
        # Step 1: Generator produces reasoning trajectory and proposes an answer
        trajectory = self.generator.generate_trajectory(
            question=question,
            choices=choices,
            image=image,
            playbook=self.playbook,
        )

        # Use the Generator's proposed answer as the adaptation prediction.
        # Fall back to (UNKNOWN) only if the Generator failed to parse JSON.
        adapted_pred = trajectory.get("proposed_choice", "") or "(UNKNOWN)"

        # Guard: if Generator produced an empty trajectory (JSON parse failure),
        # skip the Reflector/Curator cycle entirely to avoid corrupting the
        # playbook with hallucinated attributions.
        if not trajectory.get("reasoning_trajectory") and not trajectory.get("visual_observations"):
            print(f"[ACE] Warning: Empty Generator trajectory at step {step_index}. Skipping Reflector/Curator.")
            return {
                "trajectory": trajectory,
                "solver_prediction": adapted_pred,
                "solver_raw_output": "",
                "reflection": {"critique": "Skipped — empty trajectory", "helpful_bullet_ids": [], "harmful_bullet_ids": [], "delta_candidates": []},
                "curation_report": {"helpful_ids_updated": [], "harmful_ids_updated": [], "added_bullet_ids": [], "updated_bullet_ids": [], "reinforced_bullet_ids": [], "suppressed_bullet_ids": [], "operations": [], "active_bullets_count": len([b for b in self.playbook.bullets.values() if not b.is_suppressed()]), "total_bullets_count": len(self.playbook.bullets), "playbook_version": self.playbook.version, "end_of_step_summary": []},
                "playbook_version": self.playbook.version,
                "total_bullets": len(self.playbook.bullets),
            }

        # Step 2: Reflector critiques trajectory with ground truth signal
        reflection = self.reflector.reflect(
            question=question,
            choices=choices,
            trajectory=trajectory,
            solver_prediction=adapted_pred,
            solver_raw_output=trajectory.get("reasoning_trajectory", ""),
            image=image,
            playbook=self.playbook,
            ground_truth=ground_truth,
        )

        # Step 3: Curator deterministically merges delta updates into playbook
        curation_report = self.curator.curate(
            playbook=self.playbook,
            reflection=reflection,
            step_index=step_index,
        )

        return {
            "trajectory": trajectory,
            "solver_prediction": adapted_pred,
            "solver_raw_output": trajectory.get("reasoning_trajectory", ""),
            "reflection": reflection,
            "curation_report": curation_report,
            "playbook_version": self.playbook.version,
            "total_bullets": len(self.playbook.bullets),
        }

    def solve_held_out(
        self,
        question: str,
        choices: List[str],
        image: Optional[Image.Image] = None,
    ) -> Dict[str, Any]:
        """
        Solve a held-out test instance using the frozen learned playbook.
        No playbook modifications occur during test evaluation.
        """
        skill_context = self.playbook.format_for_prompt() if not self.playbook.is_empty() else None
        return self.solver.solve(
            image=image,
            question=question,
            choices=choices,
            mode="ace",
            skill=skill_context,
        )

    def save_playbook(self, json_filepath: str):
        """Save the evolved playbook to disk (both JSON and Markdown)."""
        self.playbook.save(json_filepath)
