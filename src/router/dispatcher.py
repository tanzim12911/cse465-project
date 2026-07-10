import os
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from branches.reasoning import apply_reasoning_branch
from branches.extraction import apply_extraction_branch
from branches.suppression import apply_suppression_branch
from branches.normalization import apply_normalization_branch


class BullsEyeDispatcher:
    def __init__(self, taxonomy_path=None):
        if taxonomy_path is None:
            # Resolve relative to this file so it works regardless of cwd
            taxonomy_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "taxonomy_map.json"
            )
        self.taxonomy_map = self._load_taxonomy(taxonomy_path)

    def _load_taxonomy(self, filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load taxonomy map from {filepath}. "
                  f"Defaulting all tasks to reasoning-soluble. Error: {e}")
            return {}

    def dispatch(self, task_name, image_pil, prompt):
        """
        Routes the input to the appropriate branch based on the taxonomy map.
        Returns (modified_image, modified_prompt).
        """
        if "robustness" in task_name.lower():
            return apply_normalization_branch(image_pil, prompt)

        category = self.taxonomy_map.get(task_name, "reasoning-soluble")

        if category == "perception-limited":
            return apply_extraction_branch(image_pil, prompt)
        elif category == "prior-override":
            return apply_suppression_branch(image_pil, prompt)
        else:
            # reasoning-soluble (default)
            return image_pil, apply_reasoning_branch(prompt)
