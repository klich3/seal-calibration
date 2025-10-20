"""Image processing utilities"""
import cv2
import numpy as np
from typing import Tuple, Optional


def resize_image(
    image: np.ndarray, 
    target_size: Tuple[int, int], 
    maintain_aspect: bool = True
) -> np.ndarray:
    """
    Resize image to target size
    
    Args:
        image: Input image
        target_size: Target size (width, height)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    if not maintain_aspect:
        return cv2.resize(image, target_size)
    
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create canvas and center image
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return canvas


def adjust_brightness_contrast(
    image: np.ndarray, 
    brightness: float = 0, 
    contrast: float = 0
) -> np.ndarray:
    """
    Adjust image brightness and contrast
    
    Args:
        image: Input image
        brightness: Brightness adjustment (-100 to 100)
        contrast: Contrast adjustment (-100 to 100)
        
    Returns:
        Adjusted image
    """
    # Convert to float
    img_float = image.astype(np.float32)
    
    # Apply contrast
    if contrast != 0:
        alpha = 1 + contrast / 100
        img_float = img_float * alpha
    
    # Apply brightness
    if brightness != 0:
        img_float = img_float + brightness
    
    # Clip and convert back
    img_float = np.clip(img_float, 0, 255)
    return img_float.astype(np.uint8)


def enhance_pattern_detection(gray_image: np.ndarray) -> np.ndarray:
    """
    Enhance image for better pattern detection
    
    Args:
        gray_image: Grayscale input image
        
    Returns:
        Enhanced image
    """
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_image)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    return denoised


def create_thumbnail(
    image: np.ndarray, 
    max_size: int = 200
) -> np.ndarray:
    """
    Create thumbnail of image
    
    Args:
        image: Input image
        max_size: Maximum dimension size
        
    Returns:
        Thumbnail image
    """
    h, w = image.shape[:2]
    scale = max_size / max(h, w)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return cv2.resize(image, (new_w, new_h))


def stack_images_horizontal(images: list, spacing: int = 10) -> np.ndarray:
    """
    Stack multiple images horizontally
    
    Args:
        images: List of images to stack
        spacing: Pixel spacing between images
        
    Returns:
        Stacked image
    """
    if not images:
        return None
    
    # Ensure all images have same height
    max_height = max(img.shape[0] for img in images)
    resized_images = []
    
    for img in images:
        if img.shape[0] < max_height:
            # Pad to max height
            pad = max_height - img.shape[0]
            img = cv2.copyMakeBorder(
                img, 0, pad, 0, 0, 
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        resized_images.append(img)
    
    # Add spacing
    if spacing > 0:
        spacer = np.zeros((max_height, spacing, 3), dtype=np.uint8)
        images_with_spacing = []
        for i, img in enumerate(resized_images):
            images_with_spacing.append(img)
            if i < len(resized_images) - 1:
                images_with_spacing.append(spacer)
        resized_images = images_with_spacing
    
    return np.hstack(resized_images)


def stack_images_vertical(images: list, spacing: int = 10) -> np.ndarray:
    """
    Stack multiple images vertically
    
    Args:
        images: List of images to stack
        spacing: Pixel spacing between images
        
    Returns:
        Stacked image
    """
    if not images:
        return None
    
    # Ensure all images have same width
    max_width = max(img.shape[1] for img in images)
    resized_images = []
    
    for img in images:
        if img.shape[1] < max_width:
            # Pad to max width
            pad = max_width - img.shape[1]
            img = cv2.copyMakeBorder(
                img, 0, 0, 0, pad, 
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        resized_images.append(img)
    
    # Add spacing
    if spacing > 0:
        spacer = np.zeros((spacing, max_width, 3), dtype=np.uint8)
        images_with_spacing = []
        for i, img in enumerate(resized_images):
            images_with_spacing.append(img)
            if i < len(resized_images) - 1:
                images_with_spacing.append(spacer)
        resized_images = images_with_spacing
    
    return np.vstack(resized_images)
