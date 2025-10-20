"""Projector calibration module"""
import cv2
import numpy as np
from typing import List, Tuple, Dict
from ..models.camera_params import CameraParams


class ProjectorCalibrator:
    """Handles projector calibration using Gray code patterns"""
    
    def __init__(
        self, 
        projector_width: int, 
        projector_height: int, 
        gray_bits: int = 10
    ):
        """
        Initialize projector calibrator
        
        Args:
            projector_width: Projector resolution width
            projector_height: Projector resolution height
            gray_bits: Number of Gray code bits
        """
        self.projector_width = projector_width
        self.projector_height = projector_height
        self.gray_bits = gray_bits
        
    def decode_gray_code(
        self, 
        images_horizontal: List[np.ndarray], 
        images_vertical: List[np.ndarray],
        threshold: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode Gray code patterns to get projector coordinates
        
        Args:
            images_horizontal: List of horizontal Gray code images
            images_vertical: List of vertical Gray code images
            threshold: Threshold for binary decoding
            
        Returns:
            Tuple of (x_coordinates, y_coordinates) in projector space
        """
        height, width = images_horizontal[0].shape[:2]
        
        # Decode horizontal patterns
        x_coords = np.zeros((height, width), dtype=np.float32)
        for i in range(len(images_horizontal) // 2):
            pattern = images_horizontal[2 * i]
            inverse = images_horizontal[2 * i + 1]
            
            diff = pattern.astype(np.int16) - inverse.astype(np.int16)
            mask = np.abs(diff) > threshold
            
            bit_value = (diff > 0).astype(np.int32)
            x_coords += mask * bit_value * (2 ** (len(images_horizontal) // 2 - 1 - i))
        
        # Decode vertical patterns
        y_coords = np.zeros((height, width), dtype=np.float32)
        for i in range(len(images_vertical) // 2):
            pattern = images_vertical[2 * i]
            inverse = images_vertical[2 * i + 1]
            
            diff = pattern.astype(np.int16) - inverse.astype(np.int16)
            mask = np.abs(diff) > threshold
            
            bit_value = (diff > 0).astype(np.int32)
            y_coords += mask * bit_value * (2 ** (len(images_vertical) // 2 - 1 - i))
        
        return x_coords, y_coords
    
    def build_correspondence_table(
        self,
        camera_points: np.ndarray,
        projector_points: np.ndarray,
        img_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Build look-up table for camera-projector correspondence
        
        Args:
            camera_points: Points in camera image space
            projector_points: Corresponding points in projector space
            img_size: Image size (width, height)
            
        Returns:
            LUT table as numpy array
        """
        lut = np.zeros((img_size[1], img_size[0], 2), dtype=np.float32)
        
        for cam_pt, proj_pt in zip(camera_points, projector_points):
            x, y = int(cam_pt[0]), int(cam_pt[1])
            if 0 <= x < img_size[0] and 0 <= y < img_size[1]:
                lut[y, x] = proj_pt
        
        return lut
    
    def calibrate_projector(
        self,
        camera_params: CameraParams,
        objpoints: List[np.ndarray],
        projector_points: List[np.ndarray]
    ) -> CameraParams:
        """
        Calibrate projector as an inverse camera
        
        Args:
            camera_params: Camera calibration parameters
            objpoints: 3D object points
            projector_points: 2D points in projector space
            
        Returns:
            CameraParams: Projector intrinsic parameters
        """
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        
        ret, K_proj, dist_proj, rvecs, tvecs = cv2.calibrateCamera(
            objpoints,
            projector_points,
            (self.projector_width, self.projector_height),
            None,
            None,
            flags=cv2.CALIB_RATIONAL_MODEL,
            criteria=criteria
        )
        
        return CameraParams(
            K=K_proj,
            dist=dist_proj,
            img_size=(self.projector_width, self.projector_height),
            rms_error=ret,
            rvecs=rvecs,
            tvecs=tvecs
        )
