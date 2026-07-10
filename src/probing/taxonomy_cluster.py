import os
import json


def assign_taxonomy(task_name, baseline_acc, probe_vision_acc, probe_llm_out_acc, threshold=0.15):
    """
    Classifies a task into one of three failure modes based on probe accuracies.

    - perception-limited : vision probe < 0.5 — signal never entered the model
    - prior-override     : vision probe high but drops >threshold at LLM output
    - reasoning-soluble  : signal survives to LLM output but model still fails zero-shot
    """
    if probe_vision_acc < 0.5:
        return "perception-limited"
    if (probe_vision_acc - probe_llm_out_acc) > threshold:
        return "prior-override"
    return "reasoning-soluble"


def generate_taxonomy(probe_results=None):
    """
    Generates and saves the taxonomy map.

    Args:
        probe_results: dict of {task: {baseline, vision, llm_out}} accuracies.
                       If None, uses the mocked values derived from the paper.
    """
    if probe_results is None:
        # Mocked values based on paper findings for Qwen2.5-VL-7B
        probe_results = {
            "Color Recognition" : {"baseline": 0.763, "vision": 0.85, "llm_out": 0.80},
            "Color Extraction"  : {"baseline": 0.490, "vision": 0.40, "llm_out": 0.35},
            "Object Recognition": {"baseline": 0.844, "vision": 0.88, "llm_out": 0.86},
            "Color Proportion"  : {"baseline": 0.475, "vision": 0.75, "llm_out": 0.50},
            "Color Comparison"  : {"baseline": 0.525, "vision": 0.80, "llm_out": 0.75},
            "Color Counting"    : {"baseline": 0.196, "vision": 0.70, "llm_out": 0.65},
            "Object Counting"   : {"baseline": 0.340, "vision": 0.75, "llm_out": 0.70},
            "Color Illusion"    : {"baseline": 0.441, "vision": 0.85, "llm_out": 0.55},
            "Color Mimicry"     : {"baseline": 0.557, "vision": 0.80, "llm_out": 0.60},
            "Color Blindness"   : {"baseline": 0.287, "vision": 0.65, "llm_out": 0.60},
            "Color Robustness"  : {"baseline": 0.744, "vision": 0.70, "llm_out": 0.65},
        }

    taxonomy_map = {}
    print("--- Empirical Taxonomy Classification ---")
    for task, accs in probe_results.items():
        category = assign_taxonomy(task, accs["baseline"], accs["vision"], accs["llm_out"])
        taxonomy_map[task] = category
        print(f"  {task:<25} -> {category}")

    # Always save next to this file so dispatcher can find it regardless of cwd
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "router", "taxonomy_map.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(taxonomy_map, f, indent=4)

    print(f"\nTaxonomy map saved to {out_path}")
    return taxonomy_map


if __name__ == "__main__":
    generate_taxonomy()
