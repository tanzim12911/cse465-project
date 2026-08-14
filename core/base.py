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
        if any(k in p for k in ["not present", "not exist", "does not exist", "which color does not"]):
            return {
                "classification": "color_negation",
                "skill": "List all visible colors in the image, then select the option that is absent.",
                "reflection": "Enforce explicit color enumeration for negation questions.",
            }
        elif any(k in p for k in ["illusion", "shadow", "checkerboard", "same color", "cylinder"]):
            return {
                "classification": "color_illusion",
                "skill": "De-contextualize the target patches: ignore cast shadows, background lighting, and surrounding tiles to compare true isolated pixel colors.",
                "reflection": "Isolate pixel patches to prevent ambient lighting bias.",
            }
        elif any(k in p for k in ["mimicry", "camouflage", "blend", "hidden animal", "camouflaged"]):
            return {
                "classification": "color_mimicry",
                "skill": "Trace morphological silhouettes, texture boundaries, and anatomical contours instead of relying on color contrast.",
                "reflection": "Focus on structural edge detection rather than chromatic similarity.",
            }
        elif any(k in p for k in ["how many unique colors", "how many colors", "number of colors"]):
            return {
                "classification": "color_counting",
                "skill": "Scan the scene systematically across a spatial grid (top-to-bottom, left-to-right) and list each distinct hue before counting.",
                "reflection": "Enumerate each unique hue sequentially across the grid.",
            }
        elif any(k in p for k in ["how many", "count the"]):
            return {
                "classification": "object_counting",
                "skill": "Locate each target object matching the specified color independently and count them sequentially.",
                "reflection": "Enumerate objects spatially before concluding total count.",
            }
        elif any(k in p for k in ["ishihara", "dot", "number in the circle", "color blindness", "plate"]):
            return {
                "classification": "color_blindness",
                "skill": "Trace global topological contours formed by chromatic dot contrast to identify the embedded digit or shape.",
                "reflection": "Focus on global shape closure rather than individual dot colors.",
            }
        elif any(k in p for k in ["proportion", "percentage", "area occupied", "most dominant"]):
            return {
                "classification": "color_proportion",
                "skill": "Decompose the image into dominant background and foreground color clusters to estimate percentage area coverage.",
                "reflection": "Separate background area from foreground object clusters.",
            }
        elif any(k in p for k in ["brighter", "darker", "more saturated", "compare"]):
            return {
                "classification": "color_comparison",
                "skill": "Isolate the compared regions and evaluate hue, brightness, and saturation independently.",
                "reflection": "Compare color properties in isolation.",
            }
        return {
            "classification": "color_recognition",
            "skill": "Focus on the target object and identify its surface color, ignoring surrounding background.",
            "reflection": "Refine visual attention strictly to the queried object.",
        }

