"""Base agent class for handling model generation and robust JSON parsing."""

import json
import re
from typing import Dict, Any, Optional
from PIL import Image

from .tools import ColorAnalysisTool


class BaseAgent:
    """Base agent providing robust model calling and structured JSON parsing."""

    def __init__(self, solver: Any = None):
        self.solver = solver

    def _get_color_stats_text(self, image: Optional[Image.Image], label: str = "image") -> Optional[str]:
        if image is None:
            return None
        try:
            tool = ColorAnalysisTool()
            return tool.format_for_prompt(image, label=label)
        except Exception as e:
            print(f"[BaseAgent Warning] Color analysis failed: {e}")
            return None

    def _call_model(
        self,
        system_prompt: str,
        user_prompt: str,
        image: Optional[Image.Image] = None,
        max_new_tokens: int = 256,
    ) -> Dict[str, Any]:
        """Generate structured JSON response using the local Qwen2.5-VL model."""
        if self.solver is not None:
            try:
                raw_text = self.solver.generate_text(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image=image,
                    max_new_tokens=max_new_tokens,
                )
                result = self._extract_json(raw_text)
                if result is not None:
                    return result
            except Exception as e:
                print(f"[BaseAgent Warning] Generation or JSON parsing failed: {e}")

        return {}

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Robustly extract and parse JSON object from model text output."""
        text_clean = text.strip()
        
        # 1. Direct JSON parse
        try:
            parsed = json.loads(text_clean)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # 2. Markdown ```json ... ``` code block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text_clean, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        # 3. Outermost curly braces { ... }
        match = re.search(r"(\{.*\})", text_clean, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        # 4. Trailing unclosed brace salvage (if truncated)
        match = re.search(r"(\{.*)", text_clean, re.DOTALL)
        if match:
            fragment = match.group(1).strip()
            if not fragment.endswith("}"):
                fragment += "}"
            try:
                parsed = json.loads(fragment)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        return None
