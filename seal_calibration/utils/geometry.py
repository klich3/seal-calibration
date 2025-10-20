"""Geometric transformation utilities"""
import cv2
import numpy as np
from typing import Tuple


def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert rotation matrix to Euler angles (XYZ convention)
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Tuple of (pitch, yaw, roll) in degrees
    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    singular = sy < 1e-6
    
    if not singular:
        pitch = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[1, 0], R[0, 0])
    else:
        pitch = np.arctan2(-R[1, 2], R[1, 1])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = 0
    
    return (
        np.degrees(pitch),
        np.degrees(yaw),
        np.degrees(roll)
    )


def euler_to_rotation_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """
    Convert Euler angles to rotation matrix
    
    Args:
        pitch: Rotation around X axis in degrees
        yaw: Rotation around Y axis in degrees
        roll: Rotation around Z axis in degrees
        
    Returns:
        3x3 rotation matrix
    """
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)
    roll_rad = np.radians(roll)
    
    # Rotation around X axis
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    
    # Rotation around Y axis
    R_y = np.array([
        [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
        [0, 1, 0],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ])
    
    # Rotation around Z axis
    R_z = np.array([
        [np.cos(roll_rad), -np.sin(roll_rad), 0],
        [np.sin(roll_rad), np.cos(roll_rad), 0],
        [0, 0, 1]
    ])
    
    return R_z @ R_y @ R_x


def rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    """
    Convert Rodrigues rotation vector to rotation matrix
    
    Args:
        rvec: Rodrigues rotation vector
        
    Returns:
        3x3 rotation matrix
    """
    R, _ = cv2.Rodrigues(rvec)
    return R


def matrix_to_rodrigues(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to Rodrigues vector
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Rodrigues rotation vector
    """
    rvec, _ = cv2.Rodrigues(R)
    return rvec


def compose_transforms(R1: np.ndarray, T1: np.ndarray, R2: np.ndarray, T2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compose two rigid transformations
    
    Args:
        R1: First rotation matrix
        T1: First translation vector
        R2: Second rotation matrix
        T2: Second translation vector
        
    Returns:
        Tuple of (R_composed, T_composed)
    """
    R_composed = R2 @ R1
    T_composed = R2 @ T1 + T2
    
    return R_composed, T_composed


def invert_transform(R: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Invert a rigid transformation
    
    Args:
        R: Rotation matrix
        T: Translation vector
        
    Returns:
        Tuple of (R_inv, T_inv)
    """
    R_inv = R.T
    T_inv = -R_inv @ T
    
    return R_inv, T_inv


def transform_points(points: np.ndarray, R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Apply rigid transformation to 3D points
    
    Args:
        points: Nx3 array of 3D points
        R: 3x3 rotation matrix
        T: 3x1 translation vector
        
    Returns:
        Transformed points
    """
    return (R @ points.T).T + T.T
