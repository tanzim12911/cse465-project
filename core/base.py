import json
import re
from typing import Dict, Any, Optional
from PIL import Image

class BaseAgent:
    """Base agent for handling ACE cognitive skill generation and reflection via local Qwen model."""

    def __init__(self, solver: Any = None):
        self.solver = solver

    def _call_model(
        self,
        system_prompt: str,
        user_prompt: str,
        image: Optional[Image.Image] = None,
    ) -> Dict[str, Any]:
        """Generate structured JSON response using the local Qwen2.5-VL model."""
        if self.solver is not None:
            try:
                raw_text = self.solver.generate_text(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image=image,
                )
                result = self._extract_json(raw_text)
                if result:
                    return self._normalize(result)
            except Exception as e:
                print(f"[ACE Agent Warning] Model generation parsing failed: {e}")

        # Heuristic fallback if model generation is empty/unparseable
        return self._heuristic_fallback(user_prompt)

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Robustly extract and parse JSON object from model output."""
        text_clean = text.strip()
        # 1. Direct JSON parse
        try:
            return json.loads(text_clean)
        except Exception:
            pass

        # 2. Markdown ```json ... ``` code block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text_clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass

        # 3. Any outermost curly braces { ... }
        match = re.search(r"(\{.*\})", text_clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass

        # 4. Key-value regex extraction fallback
        skill_m = re.search(r'"?skill"?\s*:\s*"([^"]+)"', text_clean, re.IGNORECASE)
        class_m = re.search(r'"?classification"?\s*:\s*"([^"]+)"', text_clean, re.IGNORECASE)
        refl_m = re.search(r'"?reflection"?\s*:\s*"([^"]+)"', text_clean, re.IGNORECASE)
        if skill_m:
            return {
                "skill": skill_m.group(1).strip(),
                "classification": class_m.group(1).strip() if class_m else "color_perception",
                "reflection": refl_m.group(1).strip() if refl_m else "",
            }

        return None

    @staticmethod
    def _normalize(result: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure result dictionary has standard string fields."""
        if "skills" in result and "skill" not in result:
            skills = result["skills"]
            result["skill"] = "\n".join(skills) if isinstance(skills, list) else str(skills)
        if "skill" not in result:
            result["skill"] = ""
        if "classification" not in result:
            result["classification"] = "color_perception"
        if "reflection" not in result:
            result["reflection"] = ""
        return result

    @staticmethod
    def _heuristic_fallback(prompt: str) -> Dict[str, Any]:
        """Rule-based fallback when model output cannot be parsed."""
        p = prompt.lower()
        if any(k in p for k in ["not present", "not exist", "does not exist"]):
            return {
                "classification": "color_negation",
                "skill": "List all visible colors in the image, then select the option that is absent.",
                "reflection": "Enforce explicit color enumeration for negation questions.",
            }
        return {
            "classification": "color_recognition",
            "skill": "Focus on the target object and identify its surface color, ignoring surrounding background.",
            "reflection": "Refine visual attention strictly to the queried object.",
        }
