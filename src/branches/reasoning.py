def apply_reasoning_branch(prompt):
    """
    Applies Chain-of-Thought (CoT) prompting for tasks classified as 'reasoning-soluble'.
    Instead of a direct answer, it prompts the model to break down its visual reasoning.
    """
    cot_instruction = (
        "\n\nLet's think step by step."
        "\n1. First, identify all the key objects mentioned in the options."
        "\n2. Next, carefully observe the colors of these objects in the image."
        "\n3. Finally, combine this information to deduce the correct answer."
    )
    
    # We append the CoT instruction before asking for the final option
    # Assuming prompt ends with "Answer with just the letter of the correct option."
    # We can modify it to encourage intermediate reasoning before the final answer.
    
    if "Answer with just the letter" in prompt:
        modified_prompt = prompt.replace(
            "Answer with just the letter of the correct option.",
            cot_instruction + "\nAfter your step-by-step reasoning, provide your final answer as 'Final Answer: [Option Letter]'."
        )
    else:
        modified_prompt = prompt + cot_instruction
        
    return modified_prompt
