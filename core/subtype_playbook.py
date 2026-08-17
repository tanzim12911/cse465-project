"""Per-subtype playbook manager for Color Illusion.

Instead of one monolithic playbook, each illusion question subtype gets its
own Playbook instance. At held-out time the router selects the right playbook
based on the question text and passes it to the Solver.

This addresses the core problem: a bullet learned for "horizontal bar
uniformity" questions (e.g., "isolate from background gradient") is irrelevant
or harmful for "which circle is darkest" questions, and vice versa.
"""

import os
import json
from typing import Dict, Optional, Any

from .playbook import Playbook
from .illusion_router import (
    classify_illusion_question,
    SUBTYPE_UNIFORMITY,
    SUBTYPE_COMPARISON,
    SUBTYPE_RANKING,
)

ALL_SUBTYPES = [SUBTYPE_UNIFORMITY, SUBTYPE_COMPARISON, SUBTYPE_RANKING]


class SubtypePlaybookManager:
    """
    Manages three separate Playbook instances, one per illusion question subtype.

    Usage
    -----
    manager = SubtypePlaybookManager(task="Color Illusion")
    pb = manager.get_playbook_for(question)   # routes to correct subtype
    manager.save_all(base_path)               # saves 3 JSON+MD files
    """

    def __init__(self, task: str = "Color Illusion"):
        self.task = task
        self.playbooks: Dict[str, Playbook] = {
            subtype: Playbook(task=f"{task} / {subtype}")
            for subtype in ALL_SUBTYPES
        }

    def get_playbook_for(self, question: str) -> Playbook:
        subtype = classify_illusion_question(question)
        return self.playbooks[subtype]

    def get_subtype_for(self, question: str) -> str:
        return classify_illusion_question(question)

    def is_all_empty(self) -> bool:
        return all(pb.is_empty() for pb in self.playbooks.values())

    def format_for_prompt(self, question: str) -> Optional[str]:
        pb = self.get_playbook_for(question)
        if pb.is_empty():
            return None
        return pb.format_for_prompt()

    def summary(self) -> Dict[str, Any]:
        return {
            subtype: {
                "total": len(pb.bullets),
                "active": sum(1 for b in pb.bullets.values() if not b.is_suppressed()),
                "suppressed": sum(1 for b in pb.bullets.values() if b.is_suppressed()),
            }
            for subtype, pb in self.playbooks.items()
        }

    def save_all(self, base_json_path: str):
        """Save each subtype playbook to a separate JSON+MD file.

        base_json_path: e.g. './results/playbook_color_illusion.json'
        Produces:
          playbook_color_illusion_uniformity.json
          playbook_color_illusion_comparison.json
          playbook_color_illusion_ranking.json
        """
        base = os.path.splitext(base_json_path)[0]
        for subtype, pb in self.playbooks.items():
            path = f"{base}_{subtype}.json"
            pb.save(path)

    @classmethod
    def load_all(cls, base_json_path: str, task: str = "Color Illusion") -> "SubtypePlaybookManager":
        manager = cls(task=task)
        base = os.path.splitext(base_json_path)[0]
        for subtype in ALL_SUBTYPES:
            path = f"{base}_{subtype}.json"
            if os.path.exists(path):
                manager.playbooks[subtype] = Playbook.load(path)
                print(f"[SubtypePlaybook] Loaded {subtype}: {len(manager.playbooks[subtype].bullets)} bullets from {path}")
        return manager
