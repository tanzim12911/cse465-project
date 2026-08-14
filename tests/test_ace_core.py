"""Unit tests for ACE Core components (Playbook, BaseAgent, Curator, Deduplication)."""

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
            category="morphology_rules",
            content="Inspect animal morphology, contours, and texture rather than relying on color similarity.",
            source_step=1,
        )
        self.assertIsNotNone(b1)
        self.assertEqual(len(pb.bullets), 1)
        self.assertEqual(pb.bullets[b1].helpful_count, 0)

        # Add near-duplicate candidate
        b2 = pb.add_bullet(
            category="morphology_rules",
            content="Inspect animal morphology, contours, and texture instead of relying on color similarity alone.",
            source_step=2,
            dedup_threshold=0.65,
        )
        # Should deduplicate by returning existing bullet ID and incrementing helpful count
        self.assertEqual(b2, b1)
        self.assertEqual(len(pb.bullets), 1)
        self.assertEqual(pb.bullets[b1].helpful_count, 1)

        # Add distinct bullet
        b3 = pb.add_bullet(
            category="counting_rules",
            content="When counting camouflaged objects, explicitly verify if zero targets are present.",
            source_step=3,
        )
        self.assertIsNotNone(b3)
        self.assertNotEqual(b3, b1)
        self.assertEqual(len(pb.bullets), 2)

    def test_curator_credit_and_merge(self):
        pb = Playbook(task="Color Illusion")
        b1 = pb.add_bullet(
            category="visual_attention",
            content="Isolate target patches from surrounding shadows before comparing hues.",
            source_step=1,
        )

        curator = Curator(solver=None)
        reflection = {
            "critique": "Isolating patches prevented shadow distortion.",
            "helpful_bullet_ids": [b1],
            "harmful_bullet_ids": [],
            "delta_candidates": [
                {
                    "category": "illusion_rules",
                    "content": "Ignore surrounding background brightness when evaluating chromatic contrast.",
                }
            ],
        }

        report = curator.curate(playbook=pb, reflection=reflection, step_index=2)
        self.assertIn(b1, report["helpful_ids_updated"])
        self.assertEqual(pb.bullets[b1].helpful_count, 1)
        self.assertEqual(len(report["added_bullet_ids"]), 1)
        self.assertEqual(len(pb.bullets), 2)

    def test_json_extraction_formats(self):
        # Direct JSON
        res = BaseAgent._extract_json('{"key": "value"}')
        self.assertEqual(res, {"key": "value"})

        # Markdown block
        md_text = 'Here is the result:\n```json\n{"proposed_choice": "(B)"}\n```\nDone.'
        res = BaseAgent._extract_json(md_text)
        self.assertEqual(res, {"proposed_choice": "(B)"})

        # Outermost braces with noisy text
        noisy_text = 'Thoughts... {"critique": "good", "delta_candidates": []} and trailing comments'
        res = BaseAgent._extract_json(noisy_text)
        self.assertEqual(res["critique"], "good")


if __name__ == "__main__":
    unittest.main()
