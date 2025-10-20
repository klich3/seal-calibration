"""Asymmetric circles pattern detector"""
import cv2
import numpy as np
from typing import Tuple, Optional


class CirclesDetector:
    """Detects asymmetric circle grid patterns"""
    
    def __init__(self, rows: int, cols: int, circle_spacing: float):
        """
        Initialize circles detector
        
        Args:
            rows: Number of circles (rows)
            cols: Number of circles (cols)
            circle_spacing: Distance between circle centers in mm
        """
        self.rows = rows
        self.cols = cols
        self.circle_spacing = circle_spacing
        self.pattern_size = (cols, rows)
        
        # SimpleBlobDetector parameters
        self.blob_params = cv2.SimpleBlobDetector_Params()
        self.blob_params.filterByArea = True
        self.blob_params.minArea = 10
        self.blob_params.maxArea = 5000
        self.blob_params.filterByCircularity = True
        self.blob_params.minCircularity = 0.7
        self.blob_params.filterByConvexity = True
        self.blob_params.minConvexity = 0.8
        self.blob_params.filterByInertia = True
        self.blob_params.minInertiaRatio = 0.5
        
        self.detector = cv2.SimpleBlobDetector_create(self.blob_params)
    
    def detect(self, gray_frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Detect asymmetric circle pattern in grayscale image
        
        Args:
            gray_frame: Grayscale image
            
        Returns:
            Tuple of (found, centers)
        """
        try:
            # Detect blobs
            keypoints = self.detector.detect(gray_frame)
            
            # Draw detected blobs (helps findCirclesGrid)
            img_with_keypoints = cv2.drawKeypoints(
                gray_frame,
                keypoints,
                np.array([]),
                (0, 0, 255),
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            
            # Find circle grid
            found, centers = cv2.findCirclesGrid(
                img_with_keypoints,
                self.pattern_size,
                flags=cv2.CALIB_CB_ASYMMETRIC_GRID,
                blobDetector=self.detector
            )
            
            return found, centers
            
        except Exception as e:
            print(f"[WARNING] Circle detection error: {str(e)}")
            return False, None
    
    def get_object_points(self) -> np.ndarray:
        """
        Get 3D object points for asymmetric circle grid
        
        Returns:
            Array of 3D object points
        """
        objp = np.zeros((self.rows * self.cols, 3), np.float32)
        
        # Asymmetric grid pattern
        for i in range(self.rows):
            for j in range(self.cols):
                objp[i * self.cols + j] = [
                    (2 * j + i % 2) * self.circle_spacing,
                    i * self.circle_spacing,
                    0
                ]
        
        return objp
    
    def draw_centers(
        self, 
        image: np.ndarray, 
        centers: np.ndarray, 
        found: bool
    ) -> np.ndarray:
        """
        Draw detected circle centers on image
        
        Args:
            image: Image to draw on
            centers: Detected centers
            found: Whether pattern was found
            
        Returns:
            Image with drawn centers
        """
        if found and centers is not None:
            for center in centers:
                x, y = int(center[0, 0]), int(center[0, 1])
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        
        return image
