import json
import re
from typing import Dict, List, Any, Optional
from PIL import Image

from .base import BaseAgent
from prompts.verifier import VERIFIER_DIAGNOSTIC_PROMPT


class Verifier(BaseAgent):
    """
    Independent Validation Evaluator.
    Programmatically executes the candidate skill across a private suite of real validation instances,
    computes validation accuracy, and synthesizes generic diagnostic feedback.
    """

    def evaluate_skill(
        self,
        skill: str,
        verifier_test_suite: List[Dict[str, Any]],
        solver: Any,
        meta_guidance: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Programmatically evaluates skill on private validation suite and generates diagnostic feedback.
        Boundary Guard: Cleanly decoupled from target instance question/choices/ground truth.
        """
        if not verifier_test_suite:
            return {
                "verifier_score": 1.0,
                "passed_count": 0,
                "total_tests": 0,
                "sanitized_feedback": "Validation suite empty. Continue general reasoning.",
                "diagnostics": [],
            }

        passed_count = 0
        total_tests = len(verifier_test_suite)
        test_diagnostics = []

        # 1. Programmatic evaluation over validation suite
        for item in verifier_test_suite:
            solver_res = solver.solve(
                image=item["image"],
                question=item["question"],
                choices=item["choices"],
                mode="adaptive_skills",
                skill=skill,
            )
            pred = solver_res.get("prediction", "")
            gt = item.get("answer", "")
            is_correct = (pred.strip().upper() == gt.strip().upper())
            if is_correct:
                passed_count += 1

            test_diagnostics.append({
                "task_type": item.get("type", item.get("task", "unknown")),
                "is_correct": is_correct,
            })

        # 2. Programmatic score calculation
        verifier_score = passed_count / total_tests if total_tests > 0 else 0.0

        # 3. Formulate structured summary for diagnostic synthesis
        failed_types = [d["task_type"] for d in test_diagnostics if not d["is_correct"]]
        passed_types = [d["task_type"] for d in test_diagnostics if d["is_correct"]]

        diag_summary = {
            "total_tests": total_tests,
            "passed_tests": passed_count,
            "verifier_score": round(verifier_score, 2),
            "passed_patterns": list(set(passed_types)),
            "failed_patterns": list(set(failed_types)),
        }

        # 4. Synthesize diagnostic feedback via model
        user_prompt = (
            f"Candidate Skill:\n\"{skill}\"\n\n"
            f"Programmatic Validation Results:\n{json.dumps(diag_summary, indent=2)}\n\n"
        )
        if meta_guidance:
            user_prompt += f"Meta-Guidance:\n{meta_guidance}\n\n"

        user_prompt += "Synthesize diagnostic guidance to improve the visual reasoning strategy."

        raw_response = self._call_model(VERIFIER_DIAGNOSTIC_PROMPT, user_prompt)
        feedback_text = raw_response.get("sanitized_feedback") or raw_response.get("feedback") or raw_response.get("skill") or ""

        if not feedback_text:
            if verifier_score == 1.0:
                feedback_text = "The skill performed robustly across validation tests. Maintain general visual invariance."
            else:
                feedback_text = "The skill struggled with subtle chromatic or spatial distinctions. Focus on isolating local pixel patches and checking contour boundaries."

        # 5. Sanitize feedback to guarantee boundary compliance
        sanitized_feedback = self.sanitize_feedback(feedback_text)

        return {
            "verifier_score": verifier_score,
            "passed_count": passed_count,
            "total_tests": total_tests,
            "sanitized_feedback": sanitized_feedback,
            "diagnostics": test_diagnostics,
        }

    def sanitize_feedback(self, text: str) -> str:
        """
        Defense-in-depth output filter to strip option letters '(A)-(E)',
        direct solution keywords, or accidental answer leaks.
        """
        if not text:
            return ""

        # Remove explicit option letter references: (A), (B), (C), (D), (E)
        sanitized = re.sub(r"\([A-E]\)", "[OPTION]", text, flags=re.IGNORECASE)
        sanitized = re.sub(r"\bOption\s+[A-E]\b", "an option", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"\bAnswer\s+is\s+[A-E]\b", "correct solution", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"\bGround\s+truth\b", "validation target", sanitized, flags=re.IGNORECASE)

        return sanitized.strip()
