"""ACE Playbook data structure, serialization, and deterministic update utilities."""

import os
import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class PlaybookBullet:
    """Itemized context unit adhering to ACE Section 3.1."""
    bullet_id: str
    category: str
    content: str
    helpful_count: int = 0
    harmful_count: int = 0
    refinement_count: int = 0
    source_step: Optional[int] = None

    def is_high_utility(self) -> bool:
        """Bullet has a strong positive history warranting extra protection.

        Invariant: a high-utility bullet must never be automatically suppressed
        or overwritten by a single reflection.
        """
        return self.helpful_count >= 5 and (self.helpful_count - self.harmful_count) >= 4

    def is_suppressed(self) -> bool:
        """Conservative suppression: deactivate bullet from Solver prompt only.

        Conditions (ALL must hold):
          - harmful_count - helpful_count >= 2   (net negative signal)
          - harmful_count >= 2                   (minimum evidence floor, lowered
                                                  from 3 to work within a 10–15
                                                  step adaptation budget)
          - not high-utility                     (protects e.g. 8/1 bullets)

        Suppression never deletes the bullet from persistent storage.
        """
        if self.is_high_utility():
            return False
        return (
            (self.harmful_count - self.helpful_count) >= 2
            and self.harmful_count >= 2
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlaybookBullet":
        return cls(
            bullet_id=data.get("bullet_id", ""),
            category=data.get("category", "general"),
            content=data.get("content", "").strip(),
            helpful_count=int(data.get("helpful_count", 0)),
            harmful_count=int(data.get("harmful_count", 0)),
            refinement_count=int(data.get("refinement_count", 0)),
            source_step=data.get("source_step"),
        )


class Playbook:
    """
    Persistent structured context playbook for Agentic Context Engineering (ACE).
    
    Contains itemized bullets, version tracking, and deterministic grow-and-refine
    operations (addition, refinement, counter attribution, deduplication, suppression).
    """

    def __init__(self, task: str = "general"):
        self.task: str = task
        self.version: int = 1
        self.bullets: Dict[str, PlaybookBullet] = {}
        self._next_id_counter: int = 1

    def is_empty(self) -> bool:
        return len(self.bullets) == 0

    def get_prefix(self) -> str:
        clean = re.sub(r"[^a-zA-Z]", "", self.task).upper()[:5]
        return clean if clean else "CTX"

    def add_bullet(
        self,
        category: str,
        content: str,
        source_step: Optional[int] = None,
        bullet_id: Optional[str] = None,
        dedup_threshold: float = 0.65,
        reinforce_on_dedup: bool = False,
    ) -> Optional[str]:
        """
        Add a new bullet to the playbook if not duplicate.
        Returns the assigned bullet_id if added (or matched), or None if rejected.

        reinforce_on_dedup: if True, increment helpful_count on the matched
        bullet (only pass True when the candidate comes from a successful outcome).
        """
        content_clean = content.strip()
        if not content_clean or len(content_clean) < 10:
            return None

        # Check for near-duplicate content
        for existing in self.bullets.values():
            if self._compute_similarity(content_clean, existing.content) >= dedup_threshold:
                # Only reinforce when the caller explicitly opts in (i.e. correct outcome)
                if reinforce_on_dedup:
                    existing.helpful_count += 1
                return existing.bullet_id

        if not bullet_id:
            prefix = self.get_prefix()
            bullet_id = f"{prefix}-{self._next_id_counter:03d}"
            self._next_id_counter += 1

        bullet = PlaybookBullet(
            bullet_id=bullet_id,
            category=category.strip().lower(),
            content=content_clean,
            helpful_count=0,
            harmful_count=0,
            refinement_count=0,
            source_step=source_step,
        )
        self.bullets[bullet_id] = bullet
        self.version += 1
        return bullet_id

    def update_bullet(
        self,
        bullet_id: str,
        new_content: str,
        new_category: Optional[str] = None,
        source_step: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        In-place refinement (UPDATE) of an existing bullet.
        Preserves historical helpful/harmful counts and increments refinement_count.
        """
        if bullet_id not in self.bullets:
            return None

        bullet = self.bullets[bullet_id]
        prev_content = bullet.content
        new_content_clean = new_content.strip()

        if not new_content_clean or len(new_content_clean) < 10:
            return None

        bullet.content = new_content_clean
        if new_category:
            bullet.category = new_category.strip().lower()
        bullet.refinement_count += 1
        bullet.source_step = source_step
        self.version += 1

        return {
            "bullet_id": bullet_id,
            "prev_content": prev_content,
            "new_content": new_content_clean,
            "category": bullet.category,
            "helpful_count": bullet.helpful_count,
            "harmful_count": bullet.harmful_count,
            "refinement_count": bullet.refinement_count,
        }

    def mark_helpful(self, bullet_ids: List[str]):
        """Increment helpful counter for specified bullet IDs."""
        for b_id in bullet_ids:
            if b_id in self.bullets:
                self.bullets[b_id].helpful_count += 1

    def mark_harmful(self, bullet_ids: List[str]):
        """Increment harmful counter for specified bullet IDs."""
        for b_id in bullet_ids:
            if b_id in self.bullets:
                self.bullets[b_id].harmful_count += 1

    def get_suppressed_bullet_ids(self) -> List[str]:
        """Return list of bullet IDs that are currently suppressed due to net negative utility."""
        return [b.bullet_id for b in self.bullets.values() if b.is_suppressed()]

    def format_for_prompt(self, max_active_bullets: int = 15) -> str:
        """
        Render structured active context for inclusion in Generator or Solver prompt.
        Filters out suppressed bullets and bounds active context to max_active_bullets.
        """
        active_bullets = [b for b in self.bullets.values() if not b.is_suppressed()]
        if not active_bullets:
            return ""

        # Limit to max_active_bullets, prioritizing higher net utility (helpful - harmful)
        active_bullets.sort(key=lambda b: (b.helpful_count - b.harmful_count), reverse=True)
        active_bullets = active_bullets[:max_active_bullets]

        lines = ["[ACE Context Playbook - Accumulated Domain Strategies]"]
        
        # Group by category
        categories: Dict[str, List[PlaybookBullet]] = {}
        for b in active_bullets:
            categories.setdefault(b.category, []).append(b)

        for cat, b_list in categories.items():
            lines.append(f"\n# Category: {cat.replace('_', ' ').title()}")
            for b in b_list:
                stats = f"[+ {b.helpful_count}/- {b.harmful_count}]" if (b.helpful_count or b.harmful_count) else ""
                lines.append(f"- [{b.bullet_id}] {b.content} {stats}".strip())

        return "\n".join(lines)

    def format_as_markdown(self) -> str:
        """Generate full human-readable markdown table of the playbook, including suppressed status."""
        suppressed_count = sum(1 for b in self.bullets.values() if b.is_suppressed())
        active_count = len(self.bullets) - suppressed_count

        lines = [
            f"# ACE Playbook: {self.task}",
            f"**Version:** {self.version} | **Total Bullets:** {len(self.bullets)} (Active: {active_count}, Suppressed: {suppressed_count})\n",
            "| ID | Category | Strategy / Insight | Helpful (+) | Harmful (-) | Refinements | Status | Step |",
            "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
        for b in self.bullets.values():
            status = "⚠️ Suppressed" if b.is_suppressed() else "✅ Active"
            lines.append(
                f"| `{b.bullet_id}` | {b.category} | {b.content} | {b.helpful_count} | {b.harmful_count} | {b.refinement_count} | {status} | {b.source_step or '-'} |"
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "version": self.version,
            "next_id_counter": self._next_id_counter,
            "bullets": {k: b.to_dict() for k, b in self.bullets.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Playbook":
        pb = cls(task=data.get("task", "general"))
        pb.version = int(data.get("version", 1))
        pb._next_id_counter = int(data.get("next_id_counter", 1))
        bullets_data = data.get("bullets", {})
        for k, b_data in bullets_data.items():
            pb.bullets[k] = PlaybookBullet.from_dict(b_data)
        return pb

    def save(self, json_filepath: str):
        """Save JSON representation and accompanying Markdown file."""
        os.makedirs(os.path.dirname(os.path.abspath(json_filepath)), exist_ok=True)
        with open(json_filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        md_filepath = os.path.splitext(json_filepath)[0] + ".md"
        with open(md_filepath, "w", encoding="utf-8") as f:
            f.write(self.format_as_markdown() + "\n")

    @classmethod
    def load(cls, json_filepath: str) -> "Playbook":
        if not os.path.exists(json_filepath):
            return cls()
        with open(json_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @staticmethod
    def _compute_similarity(text1: str, text2: str) -> float:
        """Token-level Jaccard similarity for lightweight deduplication on T4."""
        def tokenize(t: str) -> set:
            words = re.findall(r"\b\w{3,}\b", t.lower())
            return set(words)
        s1 = tokenize(text1)
        s2 = tokenize(text2)
        if not s1 or not s2:
            return 0.0
        intersection = len(s1.intersection(s2))
        union = len(s1.union(s2))
        return intersection / union if union > 0 else 0.0
