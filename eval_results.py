"""Evaluation and Comparative Report Script for ACE ColorBench Experiments."""

import os
import json
import glob
import argparse
from collections import defaultdict
from typing import List, Dict, Any


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL results file."""
    records = []
    if not os.path.exists(filepath):
        return records
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def analyze_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate overall and pattern-specific accuracy."""
    if not records:
        return {}

    total = len(records)
    correct = sum(1 for r in records if r.get("is_correct"))

    pattern_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    for r in records:
        q = r.get("question", "").lower()
        is_corr = r.get("is_correct", False)

        if any(k in q for k in ["not present", "not exist", "does not exist", "not in", "which color does not"]):
            pat = "Negation (NOT present)"
        elif "what color is" in q or "what color are" in q or "what is the color" in q:
            pat = "Direct Object Color"
        elif "how many" in q or "count" in q:
            pat = "Counting / Quantity"
        else:
            pat = "Perception / Illusion / Mimicry"

        pattern_stats[pat]["total"] += 1
        if is_corr:
            pattern_stats[pat]["correct"] += 1

    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total * 100) if total > 0 else 0.0,
        "patterns": {
            k: {
                "total": v["total"],
                "correct": v["correct"],
                "accuracy": (v["correct"] / v["total"] * 100) if v["total"] > 0 else 0.0,
            }
            for k, v in pattern_stats.items()
        },
    }


def compare_baseline_vs_ace(results_dir: str = "./results"):
    """Compare Baseline vs. ACE on identical held-out splits."""
    # Search recursively so results in run-1/, run-2/, etc. are all found
    heldout_files = glob.glob(
        os.path.join(results_dir, "**", "results_*_heldout.jsonl"),
        recursive=True,
    )
    # Also catch files directly in results_dir (flat layout)
    heldout_files += glob.glob(os.path.join(results_dir, "results_*_heldout.jsonl"))
    # Deduplicate while preserving order
    heldout_files = list(dict.fromkeys(heldout_files))
    if not heldout_files:
        print(f"No held-out results found under {results_dir}")
        return

    # Group by task
    task_results = defaultdict(dict)
    for fp in heldout_files:
        fname = os.path.basename(fp)
        recs = load_jsonl(fp)
        if not recs:
            continue
        stats = analyze_records(recs)
        
        if "_baseline_" in fname:
            task_name = fname.replace("results_", "").replace("_baseline_heldout.jsonl", "")
            task_results[task_name]["baseline"] = stats
        elif "_ace_" in fname:
            task_name = fname.replace("results_", "").replace("_ace_heldout.jsonl", "")
            task_results[task_name]["ace"] = stats

    print("\n" + "=" * 78)
    print("CSE465 RESEARCH REPORT: BASELINE vs. ACE ON HELD-OUT COLORBENCH")
    print("=" * 78)
    print(f"{'Task / Benchmark Split':30s} | {'Baseline (Held-Out)':20s} | {'ACE (Held-Out)':20s} | {'Delta':7s}")
    print("-" * 78)

    for task, res in task_results.items():
        base = res.get("baseline", {})
        ace_res = res.get("ace", {})

        base_str = f"{base.get('correct', 0)}/{base.get('total', 0)} ({base.get('accuracy', 0.0):.1f}%)" if base else "N/A"
        ace_str = f"{ace_res.get('correct', 0)}/{ace_res.get('total', 0)} ({ace_res.get('accuracy', 0.0):.1f}%)" if ace_res else "N/A"

        delta_str = "N/A"
        if base and ace_res:
            delta = ace_res.get("accuracy", 0.0) - base.get("accuracy", 0.0)
            delta_str = f"{delta:+.1f}%"

        clean_title = task.replace("_", " ").title()
        print(f"{clean_title:30s} | {base_str:20s} | {ace_str:20s} | {delta_str:7s}")

    print("=" * 78)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="./results")
    args = parser.parse_args()

    compare_baseline_vs_ace(args.dir)


if __name__ == "__main__":
    main()
