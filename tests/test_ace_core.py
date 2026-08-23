"""Unit tests for ACE Core components (Playbook, BaseAgent, Curator, Deduplication, In-place Update, Suppression)."""

import unittest
from core.playbook import Playbook, PlaybookBullet
from core.curator import Curator
from core.base import BaseAgent


class TestACECore(unittest.TestCase):

    def test_playbook_add_and_deduplicate(self):
        pb = Playbook(task="Color Illusion")
        self.assertTrue(pb.is_empty())

        # Add initial bullet
        b1 = pb.add_bullet(
            category="boundary_verification",
            content="Inspect object contours and texture rather than relying on color similarity.",
            source_step=1,
        )
        self.assertIsNotNone(b1)
        self.assertEqual(len(pb.bullets), 1)
        self.assertEqual(pb.bullets[b1].helpful_count, 0)
        self.assertEqual(pb.bullets[b1].refinement_count, 0)

        # Near-duplicate WITHOUT reinforce_on_dedup — helpful_count must NOT change
        b2 = pb.add_bullet(
            category="boundary_verification",
            content="Inspect object contours and texture instead of relying on color similarity alone.",
            source_step=2,
            dedup_threshold=0.65,
            reinforce_on_dedup=False,
        )
        self.assertEqual(b2, b1)
        self.assertEqual(len(pb.bullets), 1)
        self.assertEqual(pb.bullets[b1].helpful_count, 0)  # no reinforce

        # Near-duplicate WITH reinforce_on_dedup=True — helpful_count should increment
        b3 = pb.add_bullet(
            category="boundary_verification",
            content="Inspect object contours and texture instead of relying on color similarity alone.",
            source_step=3,
            dedup_threshold=0.65,
            reinforce_on_dedup=True,
        )
        self.assertEqual(b3, b1)
        self.assertEqual(len(pb.bullets), 1)
        self.assertEqual(pb.bullets[b1].helpful_count, 1)  # reinforced

        # Add distinct bullet
        b4 = pb.add_bullet(
            category="procedural_inspection",
            content="Compare target-patch luminance after isolating each patch from the surrounding background.",
            source_step=4,
        )
        self.assertIsNotNone(b4)
        self.assertNotEqual(b4, b1)
        self.assertEqual(len(pb.bullets), 2)

    def test_inplace_update_and_refinement(self):
        pb = Playbook(task="Color Illusion")
        b1 = pb.add_bullet(
            category="visual_attention",
            content="Focus on identifying uniform color across the entire image.",
            source_step=1,
        )
        pb.mark_helpful([b1])
        pb.mark_helpful([b1])
        pb.mark_harmful([b1])

        # Initial stats
        self.assertEqual(pb.bullets[b1].helpful_count, 2)
        self.assertEqual(pb.bullets[b1].harmful_count, 1)
        self.assertEqual(pb.bullets[b1].refinement_count, 0)

        # In-place refinement (UPDATE)
        update_info = pb.update_bullet(
            bullet_id=b1,
            new_content="Evaluate color variation within the target region independently from the surrounding context.",
            new_category="visual_attention",
            source_step=2,
        )

        self.assertIsNotNone(update_info)
        self.assertEqual(update_info["bullet_id"], b1)
        self.assertEqual(update_info["helpful_count"], 2)
        self.assertEqual(update_info["harmful_count"], 1)
        self.assertEqual(update_info["refinement_count"], 1)
        self.assertEqual(pb.bullets[b1].content, "Evaluate color variation within the target region independently from the surrounding context.")

    def test_conservative_suppression(self):
        pb = Playbook(task="Color Illusion")
        b1 = pb.add_bullet(category="visual_attention", content="Rule A with good utility.", source_step=1)
        b2 = pb.add_bullet(category="visual_attention", content="Rule B that causes frequent errors.", source_step=2)

        # b1: helpful=3, harmful=1 -> not suppressed
        pb.mark_helpful([b1, b1, b1])
        pb.mark_harmful([b1])
        self.assertFalse(pb.bullets[b1].is_suppressed())

        # b2: helpful=1, harmful=3 -> harmful - helpful = 2 -> suppressed!
        pb.mark_helpful([b2])
        pb.mark_harmful([b2, b2, b2])
        self.assertTrue(pb.bullets[b2].is_suppressed())

        # Check prompt formatting excludes b2
        prompt_context = pb.format_for_prompt()
        self.assertIn(b1, prompt_context)
        self.assertNotIn(b2, prompt_context)
        self.assertEqual(pb.get_suppressed_bullet_ids(), [b2])

    def test_curator_credit_and_update(self):
        pb = Playbook(task="Color Illusion")
        b1 = pb.add_bullet(
            category="visual_attention",
            content="Assume color is uniform when no visible gradient appears.",
            source_step=1,
        )

        # Pre-seed one prior harmful attribution so that after the reflection's
        # harmful credit (+1) the bullet reaches harmful_count=2, satisfying
        # the UPDATE evidence gate (requires >= 2 harmful attributions).
        pb.mark_harmful([b1])

        curator = Curator(solver=None)
        reflection = {
            "critique": "The rule was applied too broadly without checking target boundaries.",
            "helpful_bullet_ids": [],
            "harmful_bullet_ids": [b1],
            "delta_candidates": [
                {
                    "category": "visual_attention",
                    "content": "Verify color consistency within the local target patch specifically.",
                    "refines_bullet_id": b1,
                }
            ],
        }

        report = curator.curate(playbook=pb, reflection=reflection, step_index=2)
        self.assertIn(b1, report["updated_bullet_ids"])
        self.assertEqual(pb.bullets[b1].refinement_count, 1)
        self.assertEqual(pb.bullets[b1].content, "Verify color consistency within the local target patch specifically.")
        # harmful_count = 1 (pre-seeded) + 1 (this reflection's credit) = 2
        self.assertEqual(pb.bullets[b1].harmful_count, 2)

    def test_json_extraction_formats(self):
        # Direct JSON
        res = BaseAgent._extract_json('{"key": "value"}')
        self.assertEqual(res, {"key": "value"})

        # Markdown block
        md_text = 'Here is the result:\n```json\n{"proposed_choice": "(B)", "refines_bullet_id": "CTX-001"}\n```\nDone.'
        res = BaseAgent._extract_json(md_text)
        self.assertEqual(res, {"proposed_choice": "(B)", "refines_bullet_id": "CTX-001"})

    def test_option_parser_cot_and_robustness(self):
        from qwen_solver import QwenSolver
        choices = ["1", "2", "3", "0"]

        # Case 1: Mentioning option letters during CoT reasoning before concluding
        cot_output = (
            "Observation: The image contains one main camouflaged subject at (A) and another structure at (B). "
            "However, verifying morphological contours confirms only one actual organism is present.\n"
            "Final Answer: (A)"
        )
        self.assertEqual(QwenSolver._parse_option_letter(cot_output, choices=choices), "(A)")

        # Case 2: Bracketed conclusion at end
        cot_output2 = "The horizontal bar exhibits a continuous luminance gradient from left to right. Therefore, the answer is (C)."
        self.assertEqual(QwenSolver._parse_option_letter(cot_output2, choices=choices), "(C)")

        # Case 3: Direct letter output
        self.assertEqual(QwenSolver._parse_option_letter("(D)", choices=choices), "(D)")

    def test_compact_playbook_prompt_budget(self):
        pb = Playbook(task="Color Illusion")
        # Add 6 distinct bullets with varying helpful/harmful scores
        contents = [
            "Inspect morphological contours and anatomical symmetry to detect camouflaged subjects.",
            "Compare local patch luminance against the surrounding background ramp.",
            "Disregard surface coloration and trace structural limb boundaries.",
            "Assume background is uniform without inspecting edge transitions.",
            "Check for texture discontinuities separating subject from substrate.",
            "Evaluate illumination variance across different quadrants of the scene.",
        ]
        b_ids = []
        for i, c in enumerate(contents):
            b_id = pb.add_bullet(
                category="morphology",
                content=c,
                source_step=i+1,
            )
            b_ids.append(b_id)

        # Assign credit
        pb.mark_helpful([b_ids[0], b_ids[0], b_ids[0]])  # net +3
        pb.mark_helpful([b_ids[1], b_ids[1]])             # net +2
        pb.mark_helpful([b_ids[2]])                       # net +1
        pb.mark_harmful([b_ids[3]])                       # net -1 (harmful > helpful)
        pb.mark_helpful([b_ids[4]])                       # net 0 (helpful=1, harmful=1)
        pb.mark_harmful([b_ids[4]])

        # format_for_prompt should cap at max 4 bullets and exclude net negative (b_ids[3])
        prompt_text = pb.format_for_prompt(max_active_bullets=4)
        self.assertIn(b_ids[0], prompt_text)
        self.assertIn(b_ids[1], prompt_text)
        self.assertIn(b_ids[2], prompt_text)
        self.assertNotIn(b_ids[3], prompt_text)  # excluded due to net negative utility

    def test_tool_policy_and_cumulative_stats(self):
        pb = Playbook(task="Color Illusion")

        tool_id = pb.add_bullet(
            category="tool_policy",
            content="Use the measurement tool when choices are visually similar and the target region is ambiguous.",
            source_step=1,
            bullet_kind="tool",
        )
        self.assertIsNotNone(tool_id)
        self.assertEqual(pb.bullets[tool_id].kind, "tool")

        # tool usage should accumulate even when not yet added to the live set
        pb.record_tool_usage(tool_used=True, correct=True)
        pb.record_tool_usage(tool_used=False, correct=False)
        pb.record_tool_usage(tool_used=True, correct=False)

        summary = pb.get_tool_usage_summary()
        self.assertEqual(summary["used_total"], 2)
        self.assertEqual(summary["used_correct"], 1)
        self.assertEqual(summary["skipped_total"], 1)
        self.assertEqual(summary["skipped_correct"], 0)

        prompt_text = pb.format_for_prompt()
        self.assertIn("Use the measurement tool", prompt_text)

    def test_strict_verifier_requires_a_real_gain(self):
        from core.illusion_router import SurrogateVerifier

        class FakeSolver:
            def solve(self, image, question, choices, mode, skill=None):
                # The candidate rule changes this deterministic probe answer.
                return {"prediction": "(A)" if skill and "validated procedure" in skill else "(B)"}

        verifier = SurrogateVerifier(FakeSolver())
        pb = Playbook(task="Color Illusion")
        probe = [{"image": None, "question": "Do A and B match?", "choices": ["yes", "no"], "answer": "(A)"}]
        candidate = {
            "category": "procedural_inspection",
            "content": "Use the validated procedure to compare the isolated target patches.",
            "refines_bullet_id": None,
        }
        accepted, diag = verifier.should_commit_candidate(
            candidate=candidate,
            playbook=pb,
            reflection={"helpful_bullet_ids": [], "harmful_bullet_ids": []},
            probe_items=probe,
            step_index=1,
        )
        self.assertTrue(accepted)
        self.assertEqual(diag["correct_gain"], 1)


if __name__ == "__main__":
    unittest.main()
