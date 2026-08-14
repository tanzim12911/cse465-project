"""Main Pipeline: Adaptive Skill Generation with Iterative Prompting.

Supported Modes:
  baseline            : Direct VQA (no skills, single forward pass)
  ace_baseline        : Legacy Generator -> Solver -> Reflector -> Solver
  gen_verifier        : Generator <-> Verifier iterative feedback loop (Ablation Mode C)
  gen_verifier_oracle : Generator <-> Verifier with Hidden Oracle gating (Full Architecture)
"""

import os
import time
import argparse
from config import DEFAULT_OUTPUT_DIR, DEFAULT_ITERATIONS
from data_loader import ColorBenchDataLoader, IncrementalLogger, partition_task_dataset
from ace import ACE
from qwen_solver import QwenSolver


def parse_args():
    parser = argparse.ArgumentParser(description="CSE465 ColorBench Pipeline")
    parser.add_argument(
        "--task", type=str, default="Color Recognition",
        help="ColorBench task to evaluate (e.g. 'Color Recognition', 'Color Illusion')",
    )
    parser.add_argument(
        "--mode", type=str, default="gen_verifier_oracle",
        choices=["baseline", "ace_baseline", "adaptive_skills", "gen_verifier", "gen_verifier_oracle"],
        help="Experiment mode to run",
    )
    parser.add_argument(
        "--iterations", type=int, default=DEFAULT_ITERATIONS,
        help="Number of iterative prompting rounds per instance",
    )
    parser.add_argument(
        "--oracle_retries", type=int, default=2,
        help="Max Oracle retries on generalization failure (only for gen_verifier_oracle mode)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Directory for incremental JSONL result logs",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max target evaluation instances to process (None = all)",
    )
    return parser.parse_args()


def run_baseline(solver, item):
    """Direct VQA: question + image -> answer. No skills."""
    return solver.solve(
        image=item["image"],
        question=item["question"],
        choices=item["choices"],
        mode="baseline",
    )


def run_ace_baseline(ace_system, solver, item, iterations=2):
    """Legacy ACE baseline: Generator -> Solver -> Reflector -> Solver."""
    question = item["question"]
    choices = item["choices"]
    image = item["image"]

    plan = ace_system.generate_skill(question, choices, image=image)
    skill = plan.get("skill", "")
    classification = plan.get("classification", "unknown")

    result = solver.solve(image, question, choices, mode="adaptive_skills", skill=skill)
    prediction = result["prediction"]
    raw_output = result["raw_output"]

    all_iterations = [{
        "iteration": 1,
        "skill": skill,
        "prediction": prediction,
        "classification": classification,
    }]

    for i in range(2, iterations + 1):
        refined_plan = ace_system.reflect_and_refine(
            question, choices,
            prev_skill=skill,
            prev_answer=prediction,
            prev_raw_output=raw_output,
            image=image,
        )
        skill = refined_plan.get("skill", skill)
        reflection = refined_plan.get("reflection", "")
        classification = refined_plan.get("classification", classification)

        result = solver.solve(image, question, choices, mode="adaptive_skills", skill=skill)
        prediction = result["prediction"]
        raw_output = result["raw_output"]

        all_iterations.append({
            "iteration": i,
            "skill": skill,
            "prediction": prediction,
            "reflection": reflection,
            "classification": classification,
        })

    return {
        "prediction": prediction,
        "skill": skill,
        "classification": classification,
        "iterations": all_iterations,
        "raw_output": raw_output,
    }


def main():
    args = parse_args()
    mode = "ace_baseline" if args.mode == "adaptive_skills" else args.mode

    clean_task = args.task.lower().replace(" ", "_")
    output_path = os.path.join(
        args.output_dir, f"results_{clean_task}_{mode}.jsonl"
    )

    print("=" * 75)
    print(f"CSE465 ColorBench — Cognitive Skill Architecture Evaluation")
    print(f"Task: {args.task} | Mode: {mode} | Inner Iterations: {args.iterations}")
    print(f"Output: {output_path}")
    print("=" * 75)

    logger = IncrementalLogger(output_path)
    loader = ColorBenchDataLoader(task_filter=args.task)
    all_items = list(loader.stream_instances())

    print(f"[Pipeline] Loaded {len(all_items)} total instances for task '{args.task}'.")

    # Partition dataset if running G-V or G-V-O modes
    if mode in ["gen_verifier", "gen_verifier_oracle"]:
        verifier_pool, oracle_suite, eval_instances = partition_task_dataset(all_items)
        print(f"[Pipeline] Partitioned: {len(verifier_pool)} Verifier pool, {len(oracle_suite)} Oracle suite, {len(eval_instances)} Evaluation stream.")
    else:
        verifier_pool, oracle_suite = [], []
        eval_instances = all_items

    solver = QwenSolver()
    solver.load_model()
    ace_system = ACE(solver=solver) if mode != "baseline" else None

    correct, total = 0, 0
    start = time.time()

    for item in eval_instances:
        if args.limit and total >= args.limit:
            print(f"[Pipeline] Limit of {args.limit} reached.")
            break

        if item["idx"] in logger.processed_indices:
            continue

        print(f"\n[{total + 1}] Target idx={item['idx']} (ID: {item['id']}) | Q: {item['question']}")

        if mode == "gen_verifier_oracle":
            result = ace_system.run_gvo_pipeline(
                target_item=item,
                verifier_pool=verifier_pool,
                oracle_suite=oracle_suite,
                max_inner_iters=args.iterations,
                max_oracle_retries=args.oracle_retries,
            )
        elif mode == "gen_verifier":
            result = ace_system.run_gv_pipeline(
                target_item=item,
                verifier_pool=verifier_pool,
                max_inner_iters=args.iterations,
            )
        elif mode == "ace_baseline":
            result = run_ace_baseline(ace_system, solver, item, args.iterations)
        else:
            result = run_baseline(solver, item)

        pred = result["prediction"]
        gt = item["answer"]
        is_correct = pred.strip().upper() == gt.strip().upper()

        if is_correct:
            correct += 1
        total += 1

        print(f"  Prediction: {pred} | GT: {gt} | {'CORRECT' if is_correct else 'WRONG'}")
        if "oracle_verdict" in result:
            print(f"  Oracle Verdict: {result['oracle_verdict']} (Acc: {result['oracle_accuracy']:.2f}, Retries: {result.get('oracle_retries', 0)})")

        record = {
            "idx": item["idx"],
            "id": item["id"],
            "task": item["task"],
            "question": item["question"],
            "choices": item["choices"],
            "ground_truth": gt,
            "prediction": pred,
            "is_correct": is_correct,
            "mode": mode,
            "skill": result.get("skill"),
            "oracle_verdict": result.get("oracle_verdict"),
            "oracle_accuracy": result.get("oracle_accuracy"),
            "oracle_retries": result.get("oracle_retries"),
            "iterations": result.get("iterations"),
            "oracle_attempts": result.get("oracle_attempts"),
            "raw_output": result.get("raw_output", ""),
        }
        logger.log_result(record)

    elapsed = time.time() - start
    acc = (correct / total * 100) if total > 0 else 0.0

    print("\n" + "=" * 75)
    print(f"COMPLETED [{mode.upper()}] | {correct}/{total} correct ({acc:.2f}%) | {elapsed:.1f}s")
    print(f"Results saved to: {output_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
