from typing import Dict, List, Any


class Oracle:
    """
    Hidden Generalization Evaluator.
    Maintains a fixed set of 5 hidden real ColorBench instances to test if the
    best candidate skill generalizes beyond the Verifier's private suite.
    """

    def __init__(self, threshold: float = 0.80):
        self.threshold = threshold  # 0.80 = 4 out of 5 tests must pass

    def evaluate_generalization(
        self,
        candidate_skill: str,
        oracle_test_suite: List[Dict[str, Any]],
        solver: Any,
    ) -> Dict[str, Any]:
        """
        Programmatically evaluates candidate skill across hidden generalization suite.
        Boundary Guard: Returns strictly binary verdict + aggregate score.
        Never reveals individual test questions, failures, or case details.
        """
        total_tests = len(oracle_test_suite)
        if total_tests == 0:
            return {
                "verdict": "PASS",
                "oracle_accuracy": 1.0,
                "passed_count": 0,
                "total_tests": 0,
            }

        passed_count = 0
        for item in oracle_test_suite:
            solver_res = solver.solve(
                image=item["image"],
                question=item["question"],
                choices=item["choices"],
                mode="adaptive_skills",
                skill=candidate_skill,
            )
            pred = solver_res.get("prediction", "")
            gt = item.get("answer", "")
            if pred.strip().upper() == gt.strip().upper():
                passed_count += 1

        oracle_accuracy = passed_count / total_tests
        verdict = "PASS" if oracle_accuracy >= self.threshold else "FAIL"

        return {
            "verdict": verdict,
            "oracle_accuracy": oracle_accuracy,
            "passed_count": passed_count,
            "total_tests": total_tests,
        }
