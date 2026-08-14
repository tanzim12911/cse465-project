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
    ) -> Dict[str, Any]:
        """
        Execute one full ACE adaptation cycle on a single training/adaptation instance:
        1. Generator explores question with current playbook -> Trajectory.
        2. Solver attempts question conditioned on current playbook -> Intermediate Output.
        3. Reflector critiques trajectory & output -> Delta Candidates + Bullet Attribution.
        4. Curator applies delta updates -> Evolving Playbook.
        """
        # Step 1: Generator produces reasoning trajectory
        trajectory = self.generator.generate_trajectory(
            question=question,
            choices=choices,
            image=image,
            playbook=self.playbook,
        )

        # Step 2: Solver executes attempt conditioned on playbook
        solver_res = self.solver.solve(
            image=image,
            question=question,
            choices=choices,
            mode="ace",
            skill=self.playbook.format_for_prompt() if not self.playbook.is_empty() else None,
        )
        solver_pred = solver_res["prediction"]
        solver_raw = solver_res["raw_output"]

        # Step 3: Reflector critiques attempt
        reflection = self.reflector.reflect(
            question=question,
            choices=choices,
            trajectory=trajectory,
            solver_prediction=solver_pred,
            solver_raw_output=solver_raw,
            image=image,
            playbook=self.playbook,
        )

        # Step 4: Curator deterministically merges delta updates into playbook
        curation_report = self.curator.curate(
            playbook=self.playbook,
            reflection=reflection,
            step_index=step_index,
        )

        return {
            "trajectory": trajectory,
            "solver_prediction": solver_pred,
            "solver_raw_output": solver_raw,
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
