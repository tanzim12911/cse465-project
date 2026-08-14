"""Main Pipeline: Faithful ACE (Agentic Context Engineering) Adaptation & Evaluation.

Workflow:
  1. Creates/loads deterministic adaptation & held-out evaluation splits (fixed random seed).
  2. In 'ace' mode:
     - Phase 1 (Adaptation): Starts with an empty playbook, iteratively adapts on adaptation samples
       (Generator -> Solver -> Reflector -> Curator -> Persistent Playbook).
     - Phase 2 (Curation Log): Saves full evolved playbook to disk (JSON & Markdown).
     - Phase 3 (Evaluation): Evaluates the frozen learned playbook on the identical held-out test split.
  3. In 'baseline' mode:
     - Direct zero-shot VQA on the identical held-out test split.
     - Optionally evaluates adaptation samples for baseline diagnostic tracking.
"""

import os
import time
import argparse
from config import DEFAULT_OUTPUT_DIR
from data_loader import ColorBenchDataLoader, IncrementalLogger
from ace import ACE
from qwen_solver import QwenSolver


def parse_args():
    parser = argparse.ArgumentParser(description="CSE465 ColorBench ACE Pipeline")
    parser.add_argument(
        "--task", type=str, default="Color Mimicry",
        help="ColorBench task (e.g., 'Color Mimicry', 'Color Illusion', 'Color Recognition')",
    )
    parser.add_argument(
        "--mode", type=str, default="ace",
        choices=["baseline", "ace", "both"],
        help="baseline = direct VQA on held-out set; ace = adapt on Split A then evaluate on Split B; both = run baseline then ACE",
    )
    parser.add_argument(
        "--num_adaptation", type=int, default=10,
        help="Number of samples for ACE adaptation (Split A)",
    )
    parser.add_argument(
        "--num_eval", type=int, default=15,
        help="Number of held-out test samples for evaluation (Split B)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible split generation",
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Directory for incremental JSONL result logs and playbooks",
    )
    return parser.parse_args()


def run_baseline_evaluation(solver: QwenSolver, eval_items: list, output_path: str, split_name: str = "held_out"):
    """Evaluate baseline (zero-shot VQA) on specified items."""
    logger = IncrementalLogger(output_path)
    correct, total = 0, 0
    start = time.time()

    print("\n" + "=" * 70)
    print(f"[Baseline Evaluation] Split: {split_name} ({len(eval_items)} items)")
    print(f"Output: {output_path}")
    print("=" * 70)

    for item in eval_items:
        idx = item["idx"]
        if idx in logger.processed_indices:
            continue

        print(f"\n[Baseline - {split_name}] [{total + 1}/{len(eval_items)}] idx={idx} | Q: {item['question']}")
        res = solver.solve(
            image=item["image"],
            question=item["question"],
            choices=item["choices"],
            mode="baseline",
        )

        pred = res["prediction"]
        gt = item["answer"]
        is_corr = (pred == gt)
        if is_corr:
            correct += 1
        total += 1

        print(f"  Pred: {pred} | GT: {gt} | {'[CORRECT]' if is_corr else '[WRONG]'}")

        logger.log_result({
            "idx": idx,
            "id": item["id"],
            "task": item["task"],
            "split": split_name,
            "mode": "baseline",
            "question": item["question"],
            "choices": item["choices"],
            "ground_truth": gt,
            "prediction": pred,
            "is_correct": is_corr,
            "raw_output": res["raw_output"],
        })

    elapsed = time.time() - start
    acc = (correct / total * 100) if total > 0 else 0.0
    print(f"\n[Baseline Summary] {correct}/{total} correct ({acc:.2f}%) in {elapsed:.1f}s")
    return correct, total, acc


def run_ace_pipeline(
    solver: QwenSolver,
    adapt_items: list,
    eval_items: list,
    task_name: str,
    output_dir: str,
):
    """Execute complete ACE Adaptation + Held-Out Evaluation."""
    clean_task = task_name.lower().replace(" ", "_")
    os.makedirs(output_dir, exist_ok=True)

    playbook_json_path = os.path.join(output_dir, f"playbook_{clean_task}.json")
    adapt_log_path = os.path.join(output_dir, f"results_{clean_task}_ace_adaptation.jsonl")
    eval_log_path = os.path.join(output_dir, f"results_{clean_task}_ace_heldout.jsonl")

    # Initialize ACE system with empty Playbook
    ace_system = ACE(solver=solver, task=task_name)

    # ---------------------------------------------------------
    # Phase 1: ACE Adaptation on Adaptation Split
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"[ACE Phase 1: Adaptation] Adapting on {len(adapt_items)} samples...")
    print(f"Initial Playbook Bullets: {len(ace_system.playbook.bullets)}")
    print("=" * 70)

    adapt_logger = IncrementalLogger(adapt_log_path)
    for step_i, item in enumerate(adapt_items):
        idx = item["idx"]
        print(f"\n[ACE Adapt Step {step_i + 1}/{len(adapt_items)}] idx={idx} | Q: {item['question']}")

        adapt_record = ace_system.adapt_on_example(
            question=item["question"],
            choices=item["choices"],
            image=item["image"],
            step_index=step_i + 1,
        )

        solver_pred = adapt_record["solver_prediction"]
        gt = item["answer"]
        is_corr = (solver_pred == gt)

        print(f"  Attempt Pred: {solver_pred} | GT: {gt} | {'[CORRECT]' if is_corr else '[WRONG]'}")
        print(f"  Critique: {adapt_record['reflection'].get('critique', '')}")
        print(f"  Bullets Added: {adapt_record['curation_report'].get('added_bullet_ids', [])}")
        print(f"  Bullets Refined (UPDATE): {adapt_record['curation_report'].get('updated_bullet_ids', [])}")
        print(f"  Suppressed Bullets: {adapt_record['curation_report'].get('suppressed_bullet_ids', [])}")
        print(f"  Playbook Size: {adapt_record['total_bullets']} bullets (Active: {adapt_record['curation_report'].get('active_bullets_count', adapt_record['total_bullets'])}, v{adapt_record['playbook_version']})")

        adapt_logger.log_result({
            "step": step_i + 1,
            "idx": idx,
            "id": item["id"],
            "task": task_name,
            "split": "adaptation",
            "question": item["question"],
            "choices": item["choices"],
            "ground_truth": gt,
            "solver_prediction": solver_pred,
            "is_correct": is_corr,
            "trajectory": adapt_record["trajectory"],
            "reflection": adapt_record["reflection"],
            "curation_report": adapt_record["curation_report"],
            "playbook_version": adapt_record["playbook_version"],
            "total_bullets": adapt_record["total_bullets"],
        })

    # ---------------------------------------------------------
    # Phase 2: Persist Learned Playbook
    # ---------------------------------------------------------
    ace_system.save_playbook(playbook_json_path)
    print("\n" + "=" * 70)
    print(f"[ACE Phase 2: Saved Playbook] {len(ace_system.playbook.bullets)} total bullets saved.")
    print(f"JSON: {playbook_json_path}")
    print(f"Markdown: {os.path.splitext(playbook_json_path)[0]}.md")
    print("=" * 70)

    # ---------------------------------------------------------
    # Phase 3: Evaluate Frozen Playbook on Held-Out Test Set
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"[ACE Phase 3: Held-Out Evaluation] Evaluating on {len(eval_items)} unseen samples...")
    print("=" * 70)

    eval_logger = IncrementalLogger(eval_log_path)
    correct, total = 0, 0
    start = time.time()

    for item in eval_items:
        idx = item["idx"]
        if idx in eval_logger.processed_indices:
            continue

        print(f"\n[ACE Held-Out] [{total + 1}/{len(eval_items)}] idx={idx} | Q: {item['question']}")
        res = ace_system.solve_held_out(
            question=item["question"],
            choices=item["choices"],
            image=item["image"],
        )

        pred = res["prediction"]
        gt = item["answer"]
        is_corr = (pred == gt)
        if is_corr:
            correct += 1
        total += 1

        print(f"  ACE Pred: {pred} | GT: {gt} | {'[CORRECT]' if is_corr else '[WRONG]'}")

        eval_logger.log_result({
            "idx": idx,
            "id": item["id"],
            "task": task_name,
            "split": "held_out",
            "mode": "ace",
            "question": item["question"],
            "choices": item["choices"],
            "ground_truth": gt,
            "prediction": pred,
            "is_correct": is_corr,
            "raw_output": res["raw_output"],
            "playbook_version": ace_system.playbook.version,
            "total_bullets_used": len(ace_system.playbook.bullets),
        })

    elapsed = time.time() - start
    acc = (correct / total * 100) if total > 0 else 0.0
    print(f"\n[ACE Held-Out Summary] {correct}/{total} correct ({acc:.2f}%) in {elapsed:.1f}s")
    return correct, total, acc


def main():
    args = parse_args()

    print("=" * 70)
    print(f"CSE465 ColorBench — ACE Architecture Pipeline")
    print(f"Task: {args.task} | Mode: {args.mode} | Seed: {args.seed}")
    print(f"Adaptation Samples: {args.num_adaptation} | Held-Out Eval Samples: {args.num_eval}")
    print("=" * 70)

    # 1. Load Data and Deterministic Splits
    loader = ColorBenchDataLoader(task_filter=args.task)
    adapt_items, eval_items = loader.create_or_load_splits(
        num_adaptation=args.num_adaptation,
        num_eval=args.num_eval,
        seed=args.seed,
    )

    clean_task = args.task.lower().replace(" ", "_")
    baseline_eval_path = os.path.join(args.output_dir, f"results_{clean_task}_baseline_heldout.jsonl")
    baseline_adapt_path = os.path.join(args.output_dir, f"results_{clean_task}_baseline_adaptation.jsonl")

    # 2. Initialize Solver
    solver = QwenSolver()
    solver.load_model()

    # 3. Execute Selected Mode
    if args.mode in ["baseline", "both"]:
        # Run Baseline on Held-Out Split
        run_baseline_evaluation(solver, eval_items, baseline_eval_path, split_name="held_out")
        # Run Baseline on Adaptation Split for diagnostic tracking
        run_baseline_evaluation(solver, adapt_items, baseline_adapt_path, split_name="adaptation_diagnostic")

    if args.mode in ["ace", "both"]:
        # Run ACE Adaptation + Evaluation
        run_ace_pipeline(
            solver=solver,
            adapt_items=adapt_items,
            eval_items=eval_items,
            task_name=args.task,
            output_dir=args.output_dir,
        )

    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETED SUCCESSFULLY for {args.task}")
    print(f"Run 'python eval_results.py' to compare Baseline vs. ACE.")
    print("=" * 70)


if __name__ == "__main__":
    main()
