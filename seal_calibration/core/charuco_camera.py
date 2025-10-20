"""ChArUco-specific camera calibration module (following reference implementation)"""
import cv2
import numpy as np
from typing import List, Tuple
from ..models.camera_params import CameraParams


class CharucoCameraCalibrator:
    """Handles camera calibration using ChArUco boards"""
    
    def __init__(self, charuco_detector):
        """
        Initialize ChArUco camera calibrator
        
        Args:
            charuco_detector: CharucoDetector instance
        """
        self.charuco_detector = charuco_detector
        
    def calibrate(
        self, 
        all_corners: List[np.ndarray], 
        all_ids: List[np.ndarray],
        img_size: Tuple[int, int],
        initial_camera_matrix: np.ndarray = None,
        initial_dist_coeffs: np.ndarray = None,
        flags: int = cv2.CALIB_USE_INTRINSIC_GUESS
    ) -> CameraParams:
        """
        Perform camera calibration using ChArUco board (following reference code)
        
        Args:
            all_corners: List of detected ChArUco corners for each image
            all_ids: List of corner IDs for each image
            img_size: Image size (width, height)
            initial_camera_matrix: Initial camera matrix estimate
            initial_dist_coeffs: Initial distortion coefficients
            flags: Calibration flags (reference uses CALIB_USE_INTRINSIC_GUESS)
            
        Returns:
            CameraParams: Calibration parameters
        """
        # Initialize camera matrix if not provided (exactly like reference)
        if initial_camera_matrix is None:
            initial_camera_matrix = np.array([
                [1000., 0., img_size[0] / 2.],
                [0., 1000., img_size[1] / 2.],
                [0., 0., 1.]
            ])
        
        # Initialize distortion coefficients if not provided (exactly like reference)
        if initial_dist_coeffs is None:
            initial_dist_coeffs = np.zeros((5, 1))
        
        # Calibration criteria (exactly like reference)
        criteria = (cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9)
        
        # Calibrate using ChArUco-specific function (exactly like reference)
        ret, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            all_corners,
            all_ids,
            self.charuco_detector.charuco_board,
            img_size,
            initial_camera_matrix,
            initial_dist_coeffs,
            flags=flags,
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
