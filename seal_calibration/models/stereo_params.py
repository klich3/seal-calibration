"""Stereo calibration parameters"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class StereoParams:
    """Stereo calibration parameters
    
    Attributes:
        K_left: Left camera intrinsic matrix (3x3)
        dist_left: Left camera distortion coefficients
        K_right: Right camera intrinsic matrix (3x3)
        dist_right: Right camera distortion coefficients
        R: Rotation matrix between cameras (3x3)
        T: Translation vector between cameras (3x1)
        E: Essential matrix (3x3)
        F: Fundamental matrix (3x3)
        rms_error: RMS stereo reprojection error
        img_size: Image size (width, height)
    """
    K_left: np.ndarray
    dist_left: np.ndarray
    K_right: np.ndarray
    dist_right: np.ndarray
    R: np.ndarray
    T: np.ndarray
    E: np.ndarray
    F: np.ndarray
    rms_error: float
    img_size: Tuple[int, int]
    
    @property
    def baseline(self) -> float:
        """Stereo baseline (distance between cameras) in mm"""
        return np.linalg.norm(self.T)
