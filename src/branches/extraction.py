import cv2
import numpy as np
from PIL import Image

def get_dominant_colors_hsv(image_pil, k=3):
    """
    Extracts explicit HSV histograms to identify top K dominant colors using OpenCV.
    Bypasses internal perception limits by providing this as text context.
    """
    # Convert PIL Image to OpenCV format (BGR)
    img_array = np.array(image_pil)
    
    if len(img_array.shape) == 2:
        # Grayscale image, no color to extract
        return "Grayscale image, no dominant colors."
        
    # RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Reshape the image to be a list of pixels
    pixels = img_hsv.reshape((-1, 3))
    
    # Convert to float32 for k-means
    pixels = np.float32(pixels)
    
    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to uint8
    centers = np.uint8(centers)
    
    # Calculate the percentage of each cluster
    counts = np.bincount(labels.flatten())
    percentages = counts / len(pixels)
    
    color_info = []
    for idx, center in enumerate(centers):
        h, s, v = center
        percent = percentages[idx] * 100
        color_info.append(f"HSV({h}, {s}, {v}) roughly {percent:.1f}%")
        
    return ", ".join(color_info)

def apply_extraction_branch(image_pil, prompt):
    """
    Applies the Extraction Branch for 'perception-limited' tasks.
    Extracts dominant HSV colors and injects them into the prompt.
    """
    dominant_colors_text = get_dominant_colors_hsv(image_pil)
    
    injection = f"\n[Extracted Visual Context: The dominant colors in this image are {dominant_colors_text}]\n"
    
    # Inject before the options
    if "Answer with just the letter" in prompt:
        parts = prompt.rsplit("\nAnswer with just the letter", 1)
        modified_prompt = parts[0] + injection + "\nAnswer with just the letter" + (parts[1] if len(parts) > 1 else "")
    else:
        modified_prompt = prompt + injection
        
    return image_pil, modified_prompt
