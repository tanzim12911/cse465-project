"""Evaluation and Comparative Report Script."""

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
    """Calculate overall and fine-grained accuracy."""
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
        else:
            pat = "Other"

        pattern_stats[pat]["total"] += 1
        if is_corr:
            pattern_stats[pat]["correct"] += 1

        if r.get("ground_truth") == "(E)":
            pattern_stats["Option (E) No Answer"]["total"] += 1
            if is_corr:
                pattern_stats["Option (E) No Answer"]["correct"] += 1

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


def print_report(filepath: str):
    """Print detailed report for a single results file."""
    records = load_jsonl(filepath)
    if not records:
        print(f"No records in {filepath}")
        return

    mode = records[0].get("mode", "unknown")
    task = records[0].get("task", "unknown")
    m = analyze_records(records)

    print("\n" + "=" * 65)
    print(f"REPORT: {task} | Mode: {mode.upper()}")
    print("=" * 65)
    print(f"Overall: {m['correct']}/{m['total']} ({m['accuracy']:.2f}%)")
    print("-" * 65)
    print(f"{'Pattern':30s} | {'Score':15s} | {'Acc':8s}")
    print("-" * 65)
    for pat, pd in m["patterns"].items():
        print(f"{pat:30s} | {pd['correct']:4d}/{pd['total']:4d}      | {pd['accuracy']:6.2f}%")
    print("=" * 65)


def compare_all(results_dir: str):
    """Compare all experiment result files side-by-side."""
    files = glob.glob(os.path.join(results_dir, "results_*.jsonl"))
    if not files:
        print(f"No results found in {results_dir}")
        return

    print("\n" + "=" * 65)
    print("EXPERIMENT COMPARISON")
    print("=" * 65)
    print(f"{'Experiment':45s} | {'Accuracy':15s}")
    print("-" * 65)

    for fp in sorted(files):
        recs = load_jsonl(fp)
        if recs:
            m = analyze_records(recs)
            name = os.path.basename(fp).replace(".jsonl", "")
            print(f"{name:45s} | {m['correct']:3d}/{m['total']:3d} ({m['accuracy']:.2f}%)")

    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--dir", type=str, default="./results")
    args = parser.parse_args()

    if args.file:
        print_report(args.file)
    else:
        compare_all(args.dir)
