"""Main Pipeline: Adaptive Skill Generation with Iterative Prompting.

Implements the supervisor-mandated agentic architecture:
  Iteration 1: Gemini generates Skill v1 -> Qwen solves -> Answer v1
  Iteration 2: Gemini reflects on Answer v1 -> refines to Skill v2 -> Qwen re-solves -> Answer v2

Modes:
  baseline        : Direct VQA (no skills, no iteration)
  adaptive_skills : Full agentic iterative skill pipeline
"""

import os
import time
import argparse
from config import DEFAULT_OUTPUT_DIR, DEFAULT_ITERATIONS
from data_loader import ColorBenchDataLoader, IncrementalLogger
from ace import ACE
from qwen_solver import QwenSolver


def parse_args():
    parser = argparse.ArgumentParser(description="CSE465 ColorBench Pipeline")
    parser.add_argument(
        "--task", type=str, default="Color Recognition",
        help="ColorBench task to evaluate (e.g. 'Color Recognition', 'Color Illusion')",
    )
    parser.add_argument(
        "--mode", type=str, default="adaptive_skills",
        choices=["baseline", "adaptive_skills"],
        help="baseline = direct VQA, adaptive_skills = iterative skill pipeline",
    )
    parser.add_argument(
        "--iterations", type=int, default=DEFAULT_ITERATIONS,
        help="Number of iterative prompting rounds (only for adaptive_skills mode)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Directory for incremental JSONL result logs",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max questions to evaluate (None = all)",
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


def run_adaptive_skills(ace_system, solver, item, iterations=2):
    """Agentic iterative skill pipeline.

    Iteration 1: Generate skill -> Solve
    Iteration 2+: Reflect on previous answer -> Refine skill -> Re-solve
    """
    question = item["question"]
    choices = item["choices"]
    image = item["image"]

    # --- Iteration 1: Generate initial skill ---
    plan = ace_system.generate_skill(question, choices, image=image)
    skill = plan.get("skill", "")
    classification = plan.get("classification", "unknown")

    print(f"  [Iter 1] Classification: {classification}")
    print(f"  [Iter 1] Skill: {skill}")

    result = solver.solve(image, question, choices, mode="adaptive_skills", skill=skill)
    prediction = result["prediction"]
    raw_output = result["raw_output"]

    print(f"  [Iter 1] Answer: {prediction}")

    # --- Iterations 2+: Reflect & Refine ---
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

        print(f"  [Iter {i}] Reflection: {reflection}")
        print(f"  [Iter {i}] Refined Skill: {skill}")

        result = solver.solve(image, question, choices, mode="adaptive_skills", skill=skill)
        prediction = result["prediction"]
        raw_output = result["raw_output"]

        print(f"  [Iter {i}] Answer: {prediction}")

        all_iterations.append({
            "iteration": i,
            "skill": skill,
            "prediction": prediction,
            "reflection": reflection,
            "classification": classification,
        })

    # Final answer is from the last iteration
    result["classification"] = classification
    result["skill"] = skill
    result["iterations"] = all_iterations
    return result


def main():
    args = parse_args()

    clean_task = args.task.lower().replace(" ", "_")
    output_path = os.path.join(
        args.output_dir, f"results_{clean_task}_{args.mode}.jsonl"
    )

    print("=" * 70)
    print(f"CSE465 ColorBench — Adaptive Skill Generation Pipeline")
    print(f"Task: {args.task} | Mode: {args.mode} | Iterations: {args.iterations}")
    print(f"Output: {output_path}")
    print("=" * 70)

    logger = IncrementalLogger(output_path)
    loader = ColorBenchDataLoader(task_filter=args.task)
    solver = QwenSolver()
    solver.load_model()
    ace_system = ACE(solver=solver) if args.mode == "adaptive_skills" else None

    correct, total = 0, 0
    start = time.time()

    for item in loader.stream_instances():
        if args.limit and total >= args.limit:
            print(f"[Pipeline] Limit of {args.limit} reached.")
            break

        if item["idx"] in logger.processed_indices:
            continue

        print(f"\n[{total + 1}] idx={item['idx']} | Q: {item['question']}")

        if args.mode == "adaptive_skills":
            result = run_adaptive_skills(ace_system, solver, item, args.iterations)
        else:
            result = run_baseline(solver, item)

        pred = result["prediction"]
        gt = item["answer"]
        is_correct = pred == gt

        if is_correct:
            correct += 1
        total += 1

        print(f"  Final: {pred} | GT: {gt} | {'CORRECT' if is_correct else 'WRONG'}")

        record = {
            "idx": item["idx"],
            "id": item["id"],
            "task": item["task"],
            "question": item["question"],
            "choices": item["choices"],
            "ground_truth": gt,
            "prediction": pred,
            "is_correct": is_correct,
            "mode": args.mode,
            "classification": result.get("classification"),
            "skill": result.get("skill"),
            "iterations": result.get("iterations"),
            "raw_output": result["raw_output"],
        }
        logger.log_result(record)

    elapsed = time.time() - start
    acc = (correct / total * 100) if total > 0 else 0.0

    print("\n" + "=" * 70)
    print(f"DONE | {correct}/{total} correct ({acc:.2f}%) | {elapsed:.1f}s")
    print(f"Results saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
