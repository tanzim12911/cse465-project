"""Unit and Integration Tests for Information Boundary Enforcement in G-V-O Architecture."""

import os
import sys
import inspect
import unittest
from typing import Dict, Any, List
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.generator import Generator
from core.verifier import Verifier
from core.oracle import Oracle
from data_loader import partition_task_dataset
from ace import ACE


class MockSolver:
    """Mock VLM solver for fast deterministic unit testing without GPU."""

    def __init__(self, accuracy_map: Dict[str, str] = None):
        self.accuracy_map = accuracy_map or {}

    def solve(
        self,
        image: Image.Image,
        question: str,
        choices: List[str],
        mode: str = "adaptive_skills",
        skill: str = None,
    ) -> Dict[str, Any]:
        pred = self.accuracy_map.get(question, "(A)")
        return {
            "prediction": pred,
            "raw_output": f"Mock solver output predicting {pred}",
            "mode": mode,
        }

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        image: Image.Image = None,
        max_new_tokens: int = 256,
    ) -> str:
        if "Diagnostic" in system_prompt or "Verifier" in system_prompt:
            return '{"diagnosis": "Focus on isolated patches", "sanitized_feedback": "De-contextualize color regions and ignore shadow gradients."}'
        return '{"classification": "color_illusion", "skill": "Compare true pixel values without illumination bias."}'


class TestInformationBoundaries(unittest.TestCase):

    def setUp(self):
        self.mock_solver = MockSolver()
        self.generator = Generator(solver=self.mock_solver)
        self.verifier = Verifier(solver=self.mock_solver)
        self.oracle = Oracle(threshold=0.80)
        self.ace = ACE(solver=self.mock_solver)

    def test_generator_signature_boundaries(self):
        """Verify Generator methods NEVER accept ground truth or test suites."""
        init_sig = inspect.signature(self.generator.generate_initial_skill)
        refine_sig = inspect.signature(self.generator.refine_skill_with_feedback)

        # Assert no forbidden parameters
        forbidden = {"ground_truth", "gt", "answer", "test_suite", "tests", "oracle", "raw_output"}
        self.assertTrue(forbidden.isdisjoint(set(init_sig.parameters.keys())))
        self.assertTrue(forbidden.isdisjoint(set(refine_sig.parameters.keys())))

        # Assert required parameter presence
        self.assertIn("question", init_sig.parameters)
        self.assertIn("choices", init_sig.parameters)
        self.assertIn("verifier_feedback", refine_sig.parameters)

    def test_verifier_signature_boundaries(self):
        """Verify Verifier evaluation interface is cleanly decoupled from target question/choices."""
        eval_sig = inspect.signature(self.verifier.evaluate_skill)

        # Target question/choices/ground_truth must NOT be in evaluate_skill interface
        forbidden = {"target_question", "target_choices", "target_ground_truth", "target_gt", "target_raw_output"}
        self.assertTrue(forbidden.isdisjoint(set(eval_sig.parameters.keys())))

        self.assertIn("skill", eval_sig.parameters)
        self.assertIn("verifier_test_suite", eval_sig.parameters)

    def test_verifier_programmatic_score_calculation(self):
        """Verify Verifier computes score purely via Python arithmetic, not LLM."""
        test_suite = [
            {"question": "Q1", "choices": ["(A)", "(B)"], "answer": "(A)", "image": None, "task": "illusion"},
            {"question": "Q2", "choices": ["(A)", "(B)"], "answer": "(B)", "image": None, "task": "illusion"},
            {"question": "Q3", "choices": ["(A)", "(B)"], "answer": "(A)", "image": None, "task": "illusion"},
            {"question": "Q4", "choices": ["(A)", "(B)"], "answer": "(B)", "image": None, "task": "illusion"},
        ]
        # Solver predicts (A) for all -> Q1 correct, Q2 wrong, Q3 correct, Q4 wrong -> 2/4 = 0.5
        eval_res = self.verifier.evaluate_skill(
            skill="Test skill",
            verifier_test_suite=test_suite,
            solver=self.mock_solver,
        )
        self.assertEqual(eval_res["passed_count"], 2)
        self.assertEqual(eval_res["total_tests"], 4)
        self.assertEqual(eval_res["verifier_score"], 0.5)

    def test_verifier_feedback_sanitization(self):
        """Verify feedback sanitizer strips explicit option letters, answers, and leaks."""
        leaky_text = "The answer is (B). Choose Option C because Ground truth is red."
        sanitized = self.verifier.sanitize_feedback(leaky_text)

        self.assertNotIn("(B)", sanitized)
        self.assertNotIn("Option C", sanitized)
        self.assertNotIn("Ground truth", sanitized)

    def test_oracle_black_box_behavior(self):
        """Verify Oracle returns only binary verdict and aggregate accuracy without revealing cases."""
        oracle_suite = [
            {"question": f"OQ{i}", "choices": ["(A)", "(B)"], "answer": "(A)", "image": None}
            for i in range(5)
        ]
        # Solver predicts (A) for all -> 5/5 = 1.0 -> PASS
        eval_pass = self.oracle.evaluate_generalization("Good skill", oracle_suite, self.mock_solver)
        self.assertEqual(eval_pass["verdict"], "PASS")
        self.assertEqual(eval_pass["oracle_accuracy"], 1.0)
        self.assertEqual(eval_pass["total_tests"], 5)

        # Solver where only 3/5 pass -> 0.60 < 0.80 -> FAIL
        failing_solver = MockSolver(accuracy_map={"OQ3": "(B)", "OQ4": "(B)"})
        eval_fail = self.oracle.evaluate_generalization("Weak skill", oracle_suite, failing_solver)
        self.assertEqual(eval_fail["verdict"], "FAIL")
        self.assertEqual(eval_fail["oracle_accuracy"], 0.60)

    def test_dataset_partition_validation(self):
        """Verify partition_task_dataset strictly validates size >= 11."""
        small_items = [{"idx": i, "question": f"Q{i}"} for i in range(10)]
        with self.assertRaises(ValueError):
            partition_task_dataset(small_items, verifier_size=5, oracle_size=5)

        valid_items = [{"idx": i, "question": f"Q{i}"} for i in range(12)]
        v_pool, o_suite, evals = partition_task_dataset(valid_items, verifier_size=5, oracle_size=5)
        self.assertEqual(len(v_pool), 5)
        self.assertEqual(len(o_suite), 5)
        self.assertEqual(len(evals), 2)
        v_indices = {x["idx"] for x in v_pool}
        o_indices = {x["idx"] for x in o_suite}
        e_indices = {x["idx"] for x in evals}
        self.assertTrue(v_indices.isdisjoint(o_indices))
        self.assertTrue(v_indices.isdisjoint(e_indices))
        self.assertTrue(o_indices.isdisjoint(e_indices))

    def test_candidate_selection_zero_target_gt(self):
        """Verify candidate selection uses only VerifierScore and deterministic non-GT tie-breaking."""
        history = [
            {"iteration": 1, "skill": "Skill 1", "verifier_score": 0.50, "target_prediction": "(A)"},
            {"iteration": 2, "skill": "A very long detailed Skill 2", "verifier_score": 0.80, "target_prediction": "(B)"},
            {"iteration": 3, "skill": "Short Skill 3", "verifier_score": 0.80, "target_prediction": "(C)"},
        ]
        best_skill, best_rec = ACE._select_best_candidate(history)
        # Short Skill 3 is chosen because it is shorter than Skill 2 (conciseness bias)
        self.assertEqual(best_skill, "Short Skill 3")
        self.assertEqual(best_rec["iteration"], 3)
        self.assertTrue(best_rec["tie_occurred"])

    def test_candidate_selection_expansion_tie_breaker(self):
        """Verify tie-breaking on unused private validation examples without target GT."""
        history = [
            {"iteration": 1, "skill": "Skill A", "verifier_score": 0.67, "passed_tests": 2, "total_verifier_tests": 3, "target_prediction": "(A)"},
            {"iteration": 2, "skill": "Skill B", "verifier_score": 0.67, "passed_tests": 2, "total_verifier_tests": 3, "target_prediction": "(B)"},
        ]
        # Verifier pool with 5 examples (3 initially used, 2 unused for tie-breaking)
        verifier_pool = [
            {"question": f"Q{i}", "choices": ["(A)", "(B)"], "answer": "(A)", "image": None}
            for i in range(5)
        ]
        # Mock solver where Skill B gets Q3 and Q4 right (predicts A), while Skill A gets them wrong
        class TieSolver:
            def solve(self, image, question, choices, mode, skill):
                if skill == "Skill B":
                    return {"prediction": "(A)"}
                return {"prediction": "(B)"}

        best_skill, best_rec = ACE._select_best_candidate(
            iteration_history=history,
            verifier_pool=verifier_pool,
            solver=TieSolver(),
            initially_evaluated_count=3,
        )
        self.assertEqual(best_skill, "Skill B")
        self.assertEqual(best_rec["target_prediction"], "(B)")
        self.assertTrue(best_rec["tie_occurred"])
        self.assertTrue(best_rec["tie_broken_via_expansion"])

    def test_full_gvo_pipeline_execution(self):
        """Integration test: Verify complete GVO orchestration flow and boundary integrity."""
        target_item = {
            "idx": 10,
            "id": 10,
            "task": "color_illusion",
            "question": "Is patch A and B same color?",
            "choices": ["(A) Yes", "(B) No"],
            "answer": "(A)",
            "image": None,
        }
        verifier_pool = [
            {"idx": i, "question": f"VQ{i}", "choices": ["(A)", "(B)"], "answer": "(A)", "image": None, "task": "illusion"}
            for i in range(5)
        ]
        oracle_suite = [
            {"idx": i+5, "question": f"OQ{i}", "choices": ["(A)", "(B)"], "answer": "(A)", "image": None}
            for i in range(5)
        ]

        result = self.ace.run_gvo_pipeline(
            target_item=target_item,
            verifier_pool=verifier_pool,
            oracle_suite=oracle_suite,
            max_inner_iters=2,
            max_oracle_retries=1,
        )

        self.assertIn("prediction", result)
        self.assertIn("skill", result)
        self.assertIn("oracle_verdict", result)
        self.assertIn("oracle_accuracy", result)
        self.assertEqual(result["oracle_verdict"], "PASS")
        self.assertEqual(len(result["iterations"]), 1)  # Early stopping on verifier_score == 1.0


if __name__ == "__main__":
    unittest.main()
