import json
import time
import urllib.request
from typing import Dict, Any

import logging

# Suppress harmless google-genai automatic function calling warning
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
logging.getLogger("google_genai").setLevel(logging.ERROR)

from config import (
    GEMINI_MODEL_ID, GEMINI_FALLBACK_MODEL_ID,
    get_gemini_api_key, GEMINI_RPM_DELAY,
)

try:
    from google import genai
    from google.genai import types
    HAS_GENAI_SDK = True
except ImportError:
    HAS_GENAI_SDK = False

class BaseAgent:
    """Base agent for handling Gemini LLM calls with fallbacks."""
    def __init__(self, api_key: str = None):
        self.api_key = api_key or get_gemini_api_key()
        if HAS_GENAI_SDK:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    def _call_gemini(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Call Gemini API via SDK or REST with fallback chain."""
        # 1. Try SDK
        if HAS_GENAI_SDK and self.client:
            for model_id in [GEMINI_MODEL_ID, GEMINI_FALLBACK_MODEL_ID]:
                for attempt in range(3):
                    try:
                        config = types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            response_mime_type="application/json",
                            temperature=0.2,
                        )
                        response = self.client.models.generate_content(
                            model=model_id, contents=user_prompt, config=config,
                        )
                        result = json.loads(response.text)
                        time.sleep(GEMINI_RPM_DELAY)
                        return self._normalize(result)
                    except Exception:
                        time.sleep(2)

        # 2. Try REST endpoint
        for model_id in [GEMINI_MODEL_ID, GEMINI_FALLBACK_MODEL_ID]:
            try:
                result = self._call_rest(model_id, system_prompt, user_prompt)
                if result:
                    time.sleep(GEMINI_RPM_DELAY)
                    return self._normalize(result)
            except Exception:
                continue

        # 3. Heuristic fallback
        return self._heuristic_fallback(user_prompt)

    def _call_rest(self, model_id: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Direct REST HTTP call to Gemini API."""
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_id}:generateContent?key={self.api_key}"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.2,
            },
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data_bytes, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text_out = data["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text_out)

    @staticmethod
    def _normalize(result: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure result has 'skill' as a string (not 'skills' list)."""
        if "skills" in result and "skill" not in result:
            skills = result["skills"]
            result["skill"] = "\n".join(skills) if isinstance(skills, list) else str(skills)
        if "skill" not in result:
            result["skill"] = ""
        return result

    @staticmethod
    def _heuristic_fallback(prompt: str) -> Dict[str, Any]:
        """Rule-based fallback when API is unavailable."""
        p = prompt.lower()
        if any(k in p for k in ["not present", "not exist", "does not exist"]):
            return {
                "classification": "color_negation",
                "skill": "List all visible colors in the image, then select the option that is absent.",
            }
        return {
            "classification": "color_recognition",
            "skill": "Focus on the target object and identify its surface color, ignoring background.",
        }
