from typing import Dict, List, Any
from core import Generator, Reflector

class ACE:
    """
    Main ACE system orchestrator.
    
    Coordinates the Generator and Reflector agents.
    """
    def __init__(self, api_key: str = None):
        self.generator = Generator(api_key=api_key)
        self.reflector = Reflector(api_key=api_key)

    def generate_skill(self, question: str, choices: List[str]) -> Dict[str, Any]:
        """Pass 1: Generate initial skill."""
        return self.generator.generate_skill(question, choices)

    def reflect_and_refine(
        self,
        question: str,
        choices: List[str],
        prev_skill: str,
        prev_answer: str,
        prev_raw_output: str,
    ) -> Dict[str, Any]:
        """Pass 2+: Reflect on answer and refine skill."""
        return self.reflector.reflect_and_refine(
            question, choices, prev_skill, prev_answer, prev_raw_output
        )
