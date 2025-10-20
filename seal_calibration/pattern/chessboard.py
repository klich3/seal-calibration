"""Chessboard pattern detector"""
import cv2
import numpy as np
from typing import Tuple, Optional


class ChessboardDetector:
    """Detects chessboard calibration patterns"""
    
    def __init__(self, rows: int, cols: int, square_size: float):
        """
        Initialize chessboard detector
        
        Args:
            rows: Number of internal corners (rows)
            cols: Number of internal corners (cols)
            square_size: Size of square in mm
        """
        self.rows = rows
        self.cols = cols
        self.square_size = square_size
        self.pattern_size = (cols, rows)
        
        # Termination criteria for corner refinement
        self.criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )
    
    def detect(self, gray_frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Detect chessboard pattern in grayscale image
        
        Args:
            gray_frame: Grayscale image
            
        Returns:
            Tuple of (found, corners)
        """
        try:
            # Use EXACTLY the same method as historical script
            # NO FLAGS - just pass None to match historical behavior
            found, corners = cv2.findChessboardCorners(
                gray_frame,
                self.pattern_size,
                None
            )
            
            if found:
                # Refine corner positions with same parameters as historical
                corners = cv2.cornerSubPix(
                    gray_frame,
                    corners,
                    (11, 11),
                    (-1, -1),
                    self.criteria
                )
            
            return found, corners
            
        except Exception as e:
            print(f"[WARNING] Chessboard detection error: {str(e)}")
            return False, None
    
    def get_object_points(self) -> np.ndarray:
        """
        Get 3D object points for calibration
        
        Returns:
            Array of 3D object points
        """
        objp = np.zeros((self.rows * self.cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2)
        objp *= self.square_size
        return objp
    
    def draw_corners(
        self, 
        image: np.ndarray, 
        corners: np.ndarray, 
        found: bool
    ) -> np.ndarray:
        """
        Draw detected corners on image
        
        Args:
            image: Image to draw on
            corners: Detected corners
            found: Whether pattern was found
            
        Returns:
            Image with drawn corners
        """
        return cv2.drawChessboardCorners(
            image,
            self.pattern_size,
            corners,
            found
        )
