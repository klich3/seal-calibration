"""Camera calibration module"""
import cv2
import numpy as np
from typing import List, Tuple
from ..models.camera_params import CameraParams


class CameraCalibrator:
    """Handles individual camera calibration"""
    
    def __init__(self, pattern_detector):
        """
        Initialize camera calibrator
        
        Args:
            pattern_detector: Pattern detector instance (ChessboardDetector, etc.)
        """
        self.pattern_detector = pattern_detector
        # NO FLAGS - standard calibration returns 5 coefficients (k1, k2, p1, p2, k3)
        # RATIONAL_MODEL is applied later in stereoCalibrate to expand to 8
        self.flags = None
        
    def calibrate(
        self, 
        objpoints: List[np.ndarray], 
        imgpoints: List[np.ndarray],
        img_size: Tuple[int, int]
    ) -> CameraParams:
        """
        Perform camera calibration
        
        Args:
            objpoints: 3D points in real world space
            imgpoints: 2D points in image plane
            img_size: Image size (width, height)
            
        Returns:
            CameraParams: Calibration parameters
        """
        # Use same criteria as historical method: 30 iterations, 0.001 epsilon
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # NO FLAGS - historical method uses standard calibration (5 coefficients)
        # stereoCalibrate with RATIONAL_MODEL will expand to 8 coefficients later
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints,
            imgpoints,
            img_size,
            None,
            None,
            criteria=criteria
        )
        
        return CameraParams(
            K=K,
            dist=dist,
            img_size=img_size,
            rms_error=ret,
            rvecs=rvecs,
            tvecs=tvecs
        )
    
    def undistort(self, img: np.ndarray, params: CameraParams) -> np.ndarray:
        """
        Undistort an image using calibration parameters
        
        Args:
            img: Input image
            params: Camera calibration parameters
            
        Returns:
            Undistorted image
        """
        h, w = img.shape[:2]
        new_K, roi = cv2.getOptimalNewCameraMatrix(
            params.K, params.dist, (w, h), 1, (w, h)
        )
        
        return cv2.undistort(img, params.K, params.dist, None, new_K)
