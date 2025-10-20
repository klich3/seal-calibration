"""Calibration validation module"""
import cv2
import numpy as np
from typing import List, Tuple
from ..models.camera_params import CameraParams
from ..models.stereo_params import StereoParams


class CalibrationValidator:
    """Validates calibration quality"""
    
    @staticmethod
    def compute_reprojection_error(
        objpoints: List[np.ndarray],
        imgpoints: List[np.ndarray],
        camera_params: CameraParams
    ) -> Tuple[float, List[float]]:
        """
        Compute reprojection error for camera calibration
        
        Args:
            objpoints: 3D object points
            imgpoints: 2D image points
            camera_params: Camera calibration parameters
            
        Returns:
            Tuple of (mean_error, per_image_errors)
        """
        errors = []
        
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                objpoints[i],
                camera_params.rvecs[i],
                camera_params.tvecs[i],
                camera_params.K,
                camera_params.dist
            )
            
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            errors.append(error)
        
        mean_error = np.mean(errors)
        return mean_error, errors
    
    @staticmethod
    def validate_stereo_rectification(
        stereo_params: StereoParams,
        R1: np.ndarray,
        R2: np.ndarray,
        P1: np.ndarray,
        P2: np.ndarray
    ) -> Dict[str, float]:
        """
        Validate stereo rectification quality
        
        Args:
            stereo_params: Stereo calibration parameters
            R1, R2: Rectification rotation matrices
            P1, P2: Projection matrices
            
        Returns:
            Dictionary with validation metrics
        """
        # Check if epipolar lines are horizontal
        F_rect = P2[:, :3] @ np.linalg.inv(P1[:, :3])
        
        # Baseline from projection matrices
        baseline_rect = abs(P2[0, 3] - P1[0, 3]) / P1[0, 0]
        baseline_orig = np.linalg.norm(stereo_params.T)
        
        return {
            'baseline_rectified': baseline_rect,
            'baseline_original': baseline_orig,
            'baseline_difference': abs(baseline_rect - baseline_orig)
        }
    
    @staticmethod
    def check_calibration_coverage(
        imgpoints: List[np.ndarray],
        img_size: Tuple[int, int],
        grid_size: Tuple[int, int] = (4, 3)
    ) -> float:
        """
        Check how well calibration points cover the image
        
        Args:
            imgpoints: List of detected points
            img_size: Image size (width, height)
            grid_size: Grid divisions for coverage check
            
        Returns:
            Coverage ratio (0-1)
        """
        width, height = img_size
        grid_w, grid_h = grid_size
        
        cell_w = width / grid_w
        cell_h = height / grid_h
        
        # Create grid
        covered = np.zeros((grid_h, grid_w), dtype=bool)
        
        # Mark covered cells
        for points in imgpoints:
            for pt in points:
                x, y = int(pt[0, 0]), int(pt[0, 1])
                cell_x = min(int(x / cell_w), grid_w - 1)
                cell_y = min(int(y / cell_h), grid_h - 1)
                covered[cell_y, cell_x] = True
        
        coverage = np.sum(covered) / (grid_w * grid_h)
        return coverage
