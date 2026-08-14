from typing import Dict, List, Any, Optional
from PIL import Image
from core import Generator, Reflector

class ACE:
    """
    Main ACE System Orchestrator (Self-Improving Local Agent).

    Coordinates the Generator and Reflector using the shared local Qwen model.
    Conforms to the ACE paper (Stanford/ICLR 2026) self-improving context adaptation architecture.
    """
    def __init__(self, solver: Any = None):
        self.solver = solver
        self.generator = Generator(solver=solver)
        self.reflector = Reflector(solver=solver)

    def generate_skill(
        self,
        question: str,
        choices: List[str],
        image: Optional[Image.Image] = None,
    ) -> Dict[str, Any]:
        """Pass 1: Generate initial cognitive skill."""
        return self.generator.generate_skill(question, choices, image=image)

    def reflect_and_refine(
        self,
        question: str,
        choices: List[str],
        prev_skill: str,
        prev_answer: str,
        prev_raw_output: str,
        image: Optional[Image.Image] = None,
    ) -> Dict[str, Any]:
        """Pass 2+: Reflect on answer and refine skill."""
        return self.reflector.reflect_and_refine(
            question, choices, prev_skill, prev_answer, prev_raw_output, image=image
        )
