"""Stereo calibration module"""
import cv2
import numpy as np
from typing import List, Tuple
from ..models.stereo_params import StereoParams
from ..models.camera_params import CameraParams


class StereoCalibrator:
    """Handles stereo camera calibration"""
    
    def __init__(self, pattern_detector=None):
        """
        Initialize stereo calibrator
        
        Args:
            pattern_detector: Pattern detector instance (optional)
        """
        self.pattern_detector = pattern_detector
        # Use same flags as historical method: fix intrinsics + rational model
        self.flags = cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_RATIONAL_MODEL
        
    def calibrate(
        self, 
        objpoints: List[np.ndarray], 
        imgpoints_left: List[np.ndarray], 
        imgpoints_right: List[np.ndarray],
        camera_left: CameraParams,
        camera_right: CameraParams,
        img_size: Tuple[int, int]
    ) -> StereoParams:
        """
        Perform stereo calibration
        
        Args:
            objpoints: 3D points in real world space
            imgpoints_left: 2D points in left image plane
            imgpoints_right: 2D points in right image plane
            camera_left: Left camera calibration parameters
            camera_right: Right camera calibration parameters
            img_size: Image size (width, height)
            
        Returns:
            StereoParams: Complete stereo calibration parameters
        """
        # Use same criteria as historical method: 30 iterations, 0.001 epsilon
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # CRITICAL: Use same argument order as historical method (RIGHT-LEFT, not LEFT-RIGHT)
        # This ensures camera parameters are assigned to the correct cameras
        ret, K_right, dist_right, K_left, dist_left, R, T, E, F = cv2.stereoCalibrate(
            objpoints,
            imgpoints_right,
            imgpoints_left,
            camera_right.K,
            camera_right.dist,
            camera_left.K,
            camera_left.dist,
            img_size,
            criteria=criteria,
            flags=self.flags
        )
        
        return StereoParams(
            K_left=K_left,
            dist_left=dist_left,
            K_right=K_right,
            dist_right=dist_right,
            R=R,
            T=T,
            E=E,
            F=F,
            rms_error=ret,
            img_size=img_size
        )
    
    def rectify(
        self, 
        stereo_params: StereoParams, 
        alpha: float = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute rectification transforms for stereo pair
        
        Args:
            stereo_params: Stereo calibration parameters
            alpha: Free scaling parameter (0=no invalid pixels, 1=all pixels)
            
        Returns:
            Tuple of (R1, R2, P1, P2) rectification matrices
        """
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            stereo_params.K_left,
            stereo_params.dist_left,
            stereo_params.K_right,
            stereo_params.dist_right,
            stereo_params.img_size,
            stereo_params.R,
            stereo_params.T,
            alpha=alpha
        )
        
        return R1, R2, P1, P2
