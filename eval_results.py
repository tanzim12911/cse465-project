"""Evaluation, Comparative Reporting, and GVO Telemetry Script."""

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
    """Calculate overall, fine-grained, and GVO-specific telemetry."""
    if not records:
        return {}

    total = len(records)
    correct = sum(1 for r in records if r.get("is_correct"))

    pattern_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    task_name = (records[0].get("task") or "").lower()

    oracle_passes = sum(1 for r in records if r.get("oracle_verdict") == "PASS")
    oracle_total = sum(1 for r in records if "oracle_verdict" in r and r.get("oracle_verdict") is not None)
    total_retries = sum(r.get("oracle_retries", 0) for r in records if "oracle_retries" in r)

    for r in records:
        q = r.get("question", "").lower()
        is_corr = r.get("is_correct", False)
        gt = str(r.get("ground_truth", ""))

        if "illusion" in task_name or "illusion" in q:
            if any(w in q for w in ["shadow", "shade", "cylinder"]):
                pat = "Shadow / Lighting Illusion"
            elif any(w in q for w in ["same color", "identical", "match"]):
                pat = "Color Equality / Constancy"
            else:
                pat = "Geometric Illusion"
        elif "mimicry" in task_name or "mimicry" in q or "camouflage" in q:
            if any(w in q for w in ["animal", "insect", "spider", "fish", "frog", "bird", "snake"]):
                pat = "Fauna Camouflage"
            else:
                pat = "Object / Pattern Mimicry"
        elif "counting" in task_name or "count" in q:
            digits = [int(s) for s in q.split() if s.isdigit()]
            if digits and digits[0] <= 3:
                pat = "Low Count (<= 3)"
            else:
                pat = "High Count (> 3)"
        elif "blindness" in task_name or "ishihara" in q or "dot" in q:
            if "number" in q or "digit" in q:
                pat = "Ishihara Digit"
            else:
                pat = "Ishihara Shape/Pattern"
        else:
            if any(k in q for k in ["not present", "not exist", "does not exist", "not in", "which color does not"]):
                pat = "Negation (NOT present)"
            elif "what color is" in q or "what color are" in q or "what is the color" in q:
                pat = "Direct Object Color"
            else:
                pat = "General Identification"

        pattern_stats[pat]["total"] += 1
        if is_corr:
            pattern_stats[pat]["correct"] += 1

        if gt in ["(E)", "E", "(E) None of the above"]:
            pattern_stats["Option (E) No Answer"]["total"] += 1
            if is_corr:
                pattern_stats["Option (E) No Answer"]["correct"] += 1

    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total * 100) if total > 0 else 0.0,
        "oracle_total": oracle_total,
        "oracle_passes": oracle_passes,
        "oracle_pass_rate": (oracle_passes / oracle_total * 100) if oracle_total > 0 else None,
        "avg_oracle_retries": (total_retries / oracle_total) if oracle_total > 0 else 0.0,
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

    print("\n" + "=" * 70)
    print(f"REPORT: {task} | Mode: {mode.upper()}")
    print("=" * 70)
    print(f"Overall Accuracy: {m['correct']}/{m['total']} ({m['accuracy']:.2f}%)")
    if m["oracle_total"] > 0:
        print(f"Oracle Generalization Passes: {m['oracle_passes']}/{m['oracle_total']} ({m['oracle_pass_rate']:.2f}%)")
        print(f"Average Oracle Retries per Instance: {m['avg_oracle_retries']:.2f}")
    print("-" * 70)
    print(f"{'Pattern':35s} | {'Score':15s} | {'Acc':8s}")
    print("-" * 70)
    for pat, pd in m["patterns"].items():
        print(f"{pat:35s} | {pd['correct']:4d}/{pd['total']:4d}      | {pd['accuracy']:6.2f}%")
    print("=" * 70)


def compare_all(results_dir: str):
    """Compare all experiment result files side-by-side."""
    files = glob.glob(os.path.join(results_dir, "results_*.jsonl"))
    if not files:
        print(f"No results found in {results_dir}")
        return

    print("\n" + "=" * 75)
    print("EXPERIMENT COMPARISON MATRIX")
    print("=" * 75)
    print(f"{'Experiment':48s} | {'Accuracy':12s} | {'Oracle Pass':10s}")
    print("-" * 75)

    for fp in sorted(files):
        recs = load_jsonl(fp)
        if recs:
            m = analyze_records(recs)
            name = os.path.basename(fp).replace(".jsonl", "")
            oracle_str = f"{m['oracle_pass_rate']:.1f}%" if m["oracle_pass_rate"] is not None else "N/A"
            print(f"{name:48s} | {m['correct']:3d}/{m['total']:3d} ({m['accuracy']:5.2f}%) | {oracle_str:10s}")

    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--dir", type=str, default="./results")
    args = parser.parse_args()

    if args.file:
        print_report(args.file)
    else:
        compare_all(args.dir)
