from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
from core import Generator, Reflector, Verifier, Oracle


class ACE:
    """
    Main ACE System Orchestrator.
    Supports both legacy ACE baseline and the Generator-Verifier-Oracle (GVO) architecture.
    """

    def __init__(self, solver: Any = None):
        self.solver = solver
        self.generator = Generator(solver=solver)
        self.verifier = Verifier(solver=solver)
        self.oracle = Oracle(threshold=0.80)
        self.reflector = Reflector(solver=solver)  # Preserved for ace_baseline

    # =========================================================================
    # Legacy Baseline Interface (Preserved for ace_baseline experiments)
    # =========================================================================
    def generate_skill(
        self,
        question: str,
        choices: List[str],
        image: Optional[Image.Image] = None,
    ) -> Dict[str, Any]:
        """Pass 1 (Legacy): Generate initial cognitive skill."""
        return self.generator.generate_initial_skill(question, choices, image=image)

    def reflect_and_refine(
        self,
        question: str,
        choices: List[str],
        prev_skill: str,
        prev_answer: str,
        prev_raw_output: str,
        image: Optional[Image.Image] = None,
    ) -> Dict[str, Any]:
        """Pass 2+ (Legacy): Reflect on answer and refine skill via Reflector."""
        return self.reflector.reflect_and_refine(
            question, choices, prev_skill, prev_answer, prev_raw_output, image=image
        )

    # =========================================================================
    # Generator-Verifier-Oracle (GVO) Architecture
    # =========================================================================
    def run_gvo_pipeline(
        self,
        target_item: Dict[str, Any],
        verifier_pool: List[Dict[str, Any]],
        oracle_suite: List[Dict[str, Any]],
        max_inner_iters: int = 3,
        max_oracle_retries: int = 2,
    ) -> Dict[str, Any]:
        """
        Full G-V-O pipeline for a single target evaluation instance.

        Information Boundaries Enforced:
        1. Target GT is NEVER passed to Generator, Verifier, candidate selector, or Oracle.
        2. Generator never receives test cases, Oracle data, or GT.
        3. Verifier evaluates solely on its private validation suite.
        4. Oracle acts as a black-box generalization gate.
        """
        question = target_item["question"]
        choices = target_item["choices"]
        image = target_item["image"]

        current_test_size = 2
        active_verifier_tests = verifier_pool[:current_test_size]
        verifier_meta_guidance = None

        all_oracle_attempts = []
        best_skill = ""
        best_prediction = "(UNKNOWN)"
        oracle_verdict = "FAIL"
        oracle_acc = 0.0

        for oracle_retry in range(max_oracle_retries + 1):
            iteration_history = []
            prev_skill = None
            verifier_feedback = verifier_meta_guidance

            # 1. Inner Generator-Verifier Iterative Loop
            for k in range(1, max_inner_iters + 1):
                if k == 1:
                    gen_res = self.generator.generate_initial_skill(question, choices, image=image)
                else:
                    gen_res = self.generator.refine_skill_with_feedback(
                        question, choices, prev_skill, verifier_feedback, image=image
                    )

                candidate_skill = gen_res.get("skill", "")
                classification = gen_res.get("classification", "unknown")

                # Solve target instance with candidate skill
                target_res = self.solver.solve(
                    image=image,
                    question=question,
                    choices=choices,
                    mode="adaptive_skills",
                    skill=candidate_skill,
                )
                target_pred = target_res.get("prediction", "(UNKNOWN)")

                # Verifier evaluates skill programmatically on validation suite (NO target GT used)
                verifier_eval = self.verifier.evaluate_skill(
                    skill=candidate_skill,
                    verifier_test_suite=active_verifier_tests,
                    solver=self.solver,
                    meta_guidance=verifier_meta_guidance,
                )

                verifier_score = verifier_eval["verifier_score"]
                verifier_feedback = verifier_eval["sanitized_feedback"]

                iteration_history.append({
                    "iteration": k,
                    "skill": candidate_skill,
                    "target_prediction": target_pred,
                    "verifier_score": verifier_score,
                    "passed_tests": verifier_eval["passed_count"],
                    "total_verifier_tests": verifier_eval["total_tests"],
                    "feedback": verifier_feedback,
                    "classification": classification,
                })

                prev_skill = candidate_skill

                # Early stopping: 100% on current verifier suite (computational optimization only)
                if verifier_score == 1.0:
                    break

            # 2. Candidate Selection (Purely based on VerifierScore, zero target GT used)
            best_skill, best_record = self._select_best_candidate(iteration_history)
            best_prediction = best_record["target_prediction"]

            # 3. Oracle Hidden Generalization Evaluation (Fixed 5-example suite)
            oracle_eval = self.oracle.evaluate_generalization(
                candidate_skill=best_skill,
                oracle_test_suite=oracle_suite,
                solver=self.solver,
            )
            oracle_verdict = oracle_eval["verdict"]
            oracle_acc = oracle_eval["oracle_accuracy"]

            all_oracle_attempts.append({
                "oracle_retry": oracle_retry,
                "verifier_test_count": current_test_size,
                "oracle_verdict": oracle_verdict,
                "oracle_accuracy": oracle_acc,
                "iterations": iteration_history,
                "selected_skill": best_skill,
            })

            if oracle_verdict == "PASS":
                break

            # Oracle FAIL: Verifier validation suite was insufficient -> Strengthen Verifier
            if oracle_retry < max_oracle_retries:
                new_size = min(current_test_size + 1, len(verifier_pool))
                active_verifier_tests = verifier_pool[:new_size]
                current_test_size = new_size
                verifier_meta_guidance = (
                    "The previous evaluation suite was insufficient to guarantee generalization. "
                    "Enforce stricter visual invariance and penalize over-specialized directives."
                )

        return {
            "prediction": best_prediction,
            "skill": best_skill,
            "oracle_verdict": oracle_verdict,
            "oracle_accuracy": oracle_acc,
            "oracle_retries": len(all_oracle_attempts) - 1,
            "final_verifier_test_count": current_test_size,
            "iterations": iteration_history,
            "oracle_attempts": all_oracle_attempts,
        }

    # =========================================================================
    # Generator-Verifier (GV) Mode without Oracle Gating (for Ablation)
    # =========================================================================
    def run_gv_pipeline(
        self,
        target_item: Dict[str, Any],
        verifier_pool: List[Dict[str, Any]],
        max_inner_iters: int = 3,
    ) -> Dict[str, Any]:
        """Generator-Verifier loop without Oracle gating (Ablation Mode C)."""
        question = target_item["question"]
        choices = target_item["choices"]
        image = target_item["image"]

        active_verifier_tests = verifier_pool[:3]  # fixed standard validation suite
        iteration_history = []
        prev_skill = None
        verifier_feedback = None

        for k in range(1, max_inner_iters + 1):
            if k == 1:
                gen_res = self.generator.generate_initial_skill(question, choices, image=image)
            else:
                gen_res = self.generator.refine_skill_with_feedback(
                    question, choices, prev_skill, verifier_feedback, image=image
                )

            candidate_skill = gen_res.get("skill", "")
            classification = gen_res.get("classification", "unknown")

            target_res = self.solver.solve(
                image=image,
                question=question,
                choices=choices,
                mode="adaptive_skills",
                skill=candidate_skill,
            )
            target_pred = target_res.get("prediction", "(UNKNOWN)")

            verifier_eval = self.verifier.evaluate_skill(
                skill=candidate_skill,
                verifier_test_suite=active_verifier_tests,
                solver=self.solver,
            )

            verifier_score = verifier_eval["verifier_score"]
            verifier_feedback = verifier_eval["sanitized_feedback"]

            iteration_history.append({
                "iteration": k,
                "skill": candidate_skill,
                "target_prediction": target_pred,
                "verifier_score": verifier_score,
                "passed_tests": verifier_eval["passed_count"],
                "total_verifier_tests": verifier_eval["total_tests"],
                "feedback": verifier_feedback,
                "classification": classification,
            })

            prev_skill = candidate_skill
            if verifier_score == 1.0:
                break

        best_skill, best_record = self._select_best_candidate(iteration_history)

        return {
            "prediction": best_record["target_prediction"],
            "skill": best_skill,
            "iterations": iteration_history,
        }

    @staticmethod
    def _select_best_candidate(iteration_history: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """
        Deterministic Candidate Selection:
        Primary key: Highest VerifierScore on validation suite.
        Tie-breaker: Earliest iteration index.
        ZERO target ground truth is used.
        """
        best_record = None
        best_score = -1.0
        best_iteration = 999

        for record in iteration_history:
            score = record.get("verifier_score", 0.0)
            k = record.get("iteration", 1)

            if (score > best_score) or (score == best_score and k < best_iteration):
                best_score = score
                best_iteration = k
                best_record = record

        if best_record is None and iteration_history:
            best_record = iteration_history[-1]

        return best_record["skill"], best_record
