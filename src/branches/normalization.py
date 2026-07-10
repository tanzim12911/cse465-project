import cv2
import numpy as np
from PIL import Image

def apply_gray_world_assumption(image_pil):
    """
    Applies the Gray World Assumption for color constancy.
    This normalizes the canonical color space to stabilize robustness tasks.
    """
    img_array = np.array(image_pil)
    
    if len(img_array.shape) == 2:
        return image_pil
        
    # Calculate average values for each channel (R, G, B)
    avg_r = np.mean(img_array[:, :, 0])
    avg_g = np.mean(img_array[:, :, 1])
    avg_b = np.mean(img_array[:, :, 2])
    
    # Calculate global average
    avg_global = (avg_r + avg_g + avg_b) / 3.0
    
    # Scale each channel
    img_array[:, :, 0] = np.clip(img_array[:, :, 0] * (avg_global / avg_r), 0, 255)
    img_array[:, :, 1] = np.clip(img_array[:, :, 1] * (avg_global / avg_g), 0, 255)
    img_array[:, :, 2] = np.clip(img_array[:, :, 2] * (avg_global / avg_b), 0, 255)
    
    normalized_image = Image.fromarray(np.uint8(img_array))
    return normalized_image

def apply_normalization_branch(image_pil, prompt):
    """
    Applies the Normalization Branch for 'robustness' tasks.
    Recolors the image to a canonical color space before inference.
    """
    normalized_image = apply_gray_world_assumption(image_pil)
    
    return normalized_image, prompt
