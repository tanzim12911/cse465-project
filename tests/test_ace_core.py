"""Unit tests for ACE Core components (Playbook, BaseAgent, Curator, Deduplication, In-place Update, Suppression)."""

import unittest
from core.playbook import Playbook, PlaybookBullet
from core.curator import Curator
from core.base import BaseAgent


class TestACECore(unittest.TestCase):

    def test_playbook_add_and_deduplicate(self):
        pb = Playbook(task="Color Mimicry")
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
            category="counting_rules",
            content="When counting objects, explicitly verify if zero targets are present in the region.",
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


if __name__ == "__main__":
    unittest.main()
