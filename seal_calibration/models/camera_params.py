"""Camera calibration parameters"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class CameraParams:
    """Camera calibration parameters
    
    Attributes:
        K: Camera intrinsic matrix (3x3)
        dist: Distortion coefficients (k1, k2, p1, p2, k3, k4, k5, k6)
        img_size: Image size (width, height)
        rms_error: RMS reprojection error
        rvecs: Rotation vectors for each calibration image
        tvecs: Translation vectors for each calibration image
    """
    K: np.ndarray  # 3x3 intrinsic matrix
    dist: np.ndarray  # Distortion coefficients
    img_size: Tuple[int, int]  # (width, height)
    rms_error: float
    rvecs: list = None
    tvecs: list = None
    
    @property
    def fx(self) -> float:
        """Focal length x"""
        return self.K[0, 0]
    
    @property
    def fy(self) -> float:
        """Focal length y"""
        return self.K[1, 1]
    
    @property
    def cx(self) -> float:
        """Principal point x"""
        return self.K[0, 2]
    
    @property
    def cy(self) -> float:
        """Principal point y"""
        return self.K[1, 2]
    
    @property
    def k1(self) -> float:
        """Radial distortion k1"""
        dist_flat = self.dist.flatten()
        return float(dist_flat[0]) if len(dist_flat) > 0 else 0.0
    
    @property
    def k2(self) -> float:
        """Radial distortion k2"""
        dist_flat = self.dist.flatten()
        return float(dist_flat[1]) if len(dist_flat) > 1 else 0.0
    
    @property
    def p1(self) -> float:
        """Tangential distortion p1"""
        dist_flat = self.dist.flatten()
        return float(dist_flat[2]) if len(dist_flat) > 2 else 0.0
    
    @property
    def p2(self) -> float:
        """Tangential distortion p2"""
        dist_flat = self.dist.flatten()
        return float(dist_flat[3]) if len(dist_flat) > 3 else 0.0
    
    @property
    def k3(self) -> float:
        """Radial distortion k3"""
        dist_flat = self.dist.flatten()
        return float(dist_flat[4]) if len(dist_flat) > 4 else 0.0
    
    @property
    def k4(self) -> float:
        """Radial distortion k4 (RATIONAL_MODEL)"""
        dist_flat = self.dist.flatten()
        return float(dist_flat[5]) if len(dist_flat) > 5 else 0.0
    
    @property
    def k5(self) -> float:
        """Radial distortion k5 (RATIONAL_MODEL)"""
        dist_flat = self.dist.flatten()
        return float(dist_flat[6]) if len(dist_flat) > 6 else 0.0
    
    @property
    def k6(self) -> float:
        """Radial distortion k6 (RATIONAL_MODEL)"""
        dist_flat = self.dist.flatten()
        return float(dist_flat[7]) if len(dist_flat) > 7 else 0.0
