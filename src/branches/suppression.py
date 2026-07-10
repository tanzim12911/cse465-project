from PIL import Image


def apply_suppression_branch(image_pil, prompt):
    """
    Suppression Branch for 'prior-override' tasks (Color Illusion, Color Mimicry).

    The paper (Finding 4) shows that converting colorful images to grayscale
    improves accuracy on these tasks because the model's language priors
    ("fire is orange", "sky is blue") override what it actually sees.
    Removing color forces the model to reason from shape and luminance instead.
    """
    gray_rgb_image = image_pil.convert('L').convert('RGB')

    injection = (
        "\n[Instruction: Ignore any textual priors about color. "
        "Rely STRICTLY on the luminance and shapes in this grayscale image to answer.]\n"
    )

    if "Answer with just the letter" in prompt:
        parts = prompt.rsplit("\nAnswer with just the letter", 1)
        modified_prompt = (
            parts[0] + injection
            + "\nAnswer with just the letter"
            + (parts[1] if len(parts) > 1 else "")
        )
    else:
        modified_prompt = prompt + injection

    return gray_rgb_image, modified_prompt
