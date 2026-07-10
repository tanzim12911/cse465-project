import os
import json
import sys

# Add src to path to import branches
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from branches.reasoning import apply_reasoning_branch
from branches.extraction import apply_extraction_branch
from branches.suppression import apply_suppression_branch
from branches.normalization import apply_normalization_branch

class BullsEyeDispatcher:
    def __init__(self, taxonomy_path="./src/router/taxonomy_map.json"):
        self.taxonomy_map = self._load_taxonomy(taxonomy_path)
        
    def _load_taxonomy(self, filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load taxonomy map from {filepath}. Using fallback defaults. Error: {e}")
            return {}
            
    def dispatch(self, task_name, image_pil, prompt):
        """
        Routes the input to the appropriate branch based on the Phase 3 taxonomy mapping.
        Returns the modified (image_pil, prompt) tuple.
        """
        
        # Color robustness is a special category evaluated with color transformations
        if "robustness" in task_name.lower():
            print(f"[{task_name}] Routing to Normalization Branch")
            return apply_normalization_branch(image_pil, prompt)
            
        category = self.taxonomy_map.get(task_name, "reasoning-soluble")
        
        if category == "reasoning-soluble":
            print(f"[{task_name}] Routing to Reasoning Branch")
            new_prompt = apply_reasoning_branch(prompt)
            return image_pil, new_prompt
            
        elif category == "perception-limited":
            print(f"[{task_name}] Routing to Extraction Branch")
            return apply_extraction_branch(image_pil, prompt)
            
        elif category == "prior-override":
            print(f"[{task_name}] Routing to Suppression Branch")
            return apply_suppression_branch(image_pil, prompt)
            
        else:
            print(f"[{task_name}] Unknown category '{category}'. Defaulting to base input.")
            return image_pil, prompt
