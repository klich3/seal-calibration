"""Fixed stripe pattern detector for structured light"""
import cv2
import numpy as np
from typing import Tuple, Optional, List


class StripesDetector:
    """Detects fixed stripe patterns for structured light calibration"""
    
    def __init__(self, num_stripes: int, stripe_width: float):
        """
        Initialize stripes detector
        
        Args:
            num_stripes: Number of vertical stripes
            stripe_width: Width of each stripe in mm
        """
        self.num_stripes = num_stripes
        self.stripe_width = stripe_width
    
    def detect_edges(self, gray_frame: np.ndarray, threshold: int = 50) -> np.ndarray:
        """
        Detect stripe edges in grayscale image
        
        Args:
            gray_frame: Grayscale image
            threshold: Edge detection threshold
            
        Returns:
            Binary edge map
        """
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        
        # Detect edges using Canny
        edges = cv2.Canny(blurred, threshold, threshold * 2)
        
        return edges
    
    def extract_stripe_centers(
        self, 
        gray_frame: np.ndarray, 
        threshold: int = 50
    ) -> List[np.ndarray]:
        """
        Extract vertical stripe center lines
        
        Args:
            gray_frame: Grayscale image
            threshold: Detection threshold
            
        Returns:
            List of stripe center coordinates
        """
        height, width = gray_frame.shape
        
        # Compute horizontal gradient
        grad_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=5)
        
        # Find peaks (stripe centers)
        stripe_centers = []
        
        for y in range(height):
            row = grad_x[y, :]
            
            # Find peaks in gradient
            peaks = []
            for x in range(1, width - 1):
                if abs(row[x]) > threshold:
                    if row[x - 1] < row[x] > row[x + 1]:
                        peaks.append(x)
            
            if len(peaks) == self.num_stripes:
                stripe_centers.append(np.array(peaks))
        
        return stripe_centers
    
    def get_stripe_coordinates(
        self, 
        stripe_centers: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get 2D image coordinates and 3D object coordinates
        
        Args:
            stripe_centers: List of stripe center coordinates per row
            
        Returns:
            Tuple of (image_points, object_points)
        """
        image_points = []
        object_points = []
        
        for y, centers in enumerate(stripe_centers):
            for stripe_idx, x in enumerate(centers):
                # Image point
                image_points.append([x, y])
                
                # Object point (3D)
                object_points.append([
                    stripe_idx * self.stripe_width,
                    0,
                    0
                ])
        
        return (
            np.array(image_points, dtype=np.float32),
            np.array(object_points, dtype=np.float32)
        )
    
    def visualize_stripes(
        self, 
        image: np.ndarray, 
        stripe_centers: List[np.ndarray]
    ) -> np.ndarray:
        """
        Visualize detected stripe centers
        
        Args:
            image: Image to draw on
            stripe_centers: Detected stripe centers
            
        Returns:
            Image with visualized stripes
        """
        vis_image = image.copy()
        
        for y, centers in enumerate(stripe_centers):
            for x in centers:
                cv2.circle(vis_image, (int(x), y), 2, (0, 255, 0), -1)
        
        return vis_image
