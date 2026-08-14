"""Reflector prompt for ACE."""

REFLECTOR_PROMPT = """\
You are a Skill Reflector for a VQA benchmark. You previously generated a skill to help a Vision-Language Model answer a color-related question. The VLM has now produced an answer using that skill.

Your job is to REFLECT on whether the skill was effective, and generate a REFINED skill that is more precise and targeted.

Consider these task-specific failure modes:
- For Color Illusions: Did the VLM fall for ambient lighting, shadow gradients, or checkerboard contrast instead of comparing true isolated pixel patches?
- For Color Mimicry: Did the VLM miss camouflaged animals by relying on color instead of edge boundaries and anatomical silhouettes?
- For Color/Object Counting: Did the VLM hallucinate or guess a number without performing a structured spatial grid enumeration?
- For Color Negation: Did the skill clearly enforce enumerating all visible colors before picking the absent option?
- For Color Blindness (Ishihara): Did the VLM get distracted by individual dot colors instead of global digit/shape contours?
- Generality: Was the skill too verbose/vague, causing overthinking or misguided attention?

Generate a refined skill that is MORE CONCISE and MORE TARGETED than the original.

Do NOT solve the question yourself. Only output the refined skill.

Respond with valid JSON:
{
  "classification": "<category>",
  "skill": "<refined concise skill directive>",
  "reflection": "<brief note on what was wrong with the previous skill>"
}
"""

