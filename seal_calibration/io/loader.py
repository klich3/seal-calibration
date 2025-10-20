"""SEAL calibration file loader"""
import numpy as np
import re
from typing import Tuple, Dict, Optional
from ..models.seal_calib import SEALCalibration
from ..models.camera_params import CameraParams
from ..models.stereo_params import StereoParams


class SEALCalibrationLoader:
    """Loads and parses SEAL calibration files"""
    
    @staticmethod
    def load(filepath: str) -> SEALCalibration:
        """
        Load calibration from SEAL format file
        
        Args:
            filepath: Path to SEAL calibration file
            
        Returns:
            SEALCalibration object with all parameters
        """
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [line.strip() for line in f.readlines()]
        
        if len(lines) < 10:
            raise ValueError(f"Invalid SEAL calibration file: too few lines ({len(lines)})")
        
        # Line 1: Resolution (W H)
        resolution = tuple(map(int, lines[0].split()))
        
        # Line 2: Scale factors (baseline_scale, depth_scale) - FACTORY PARAMS
        scale_factors = tuple(map(float, lines[1].split()))
        
        # Line 3: Offset center (dx, dy) - FACTORY PARAMS
        offset_center = tuple(map(int, lines[2].split()))
        
        # Line 4: Offset tilt (tilt, height) - FACTORY PARAMS
        offset_tilt = tuple(map(int, lines[3].split()))
        
        # Lines 5-7: Left camera intrinsics (single line format)
        left_params = SEALCalibrationLoader._parse_camera_params(
            [lines[4]], resolution  # Pass single line as list
        )
        
        # Lines 8-10: Right camera intrinsics (single line format)
        right_params = SEALCalibrationLoader._parse_camera_params(
            [lines[5]], resolution  # Pass single line as list
        )
        
        # Parse LUT table if present (lines after camera params until metadata)
        lut_table = None
        metadata_idx = len(lines) - 1
        
        # Find metadata line (starts with ***)
        for i in range(len(lines) - 1, 5, -1):  # Start from line 6 (after cameras)
            if lines[i].startswith("***DevID:"):
                metadata_idx = i
                break
        
        # Extract LUT if present (after line 7/index 6 which is projector params)
        # LUT starts after line 7 (projector) or line 6 if no projector
        lut_start_idx = 7  # After camera_left, camera_right, projector
        if metadata_idx > lut_start_idx:
            lut_lines = lines[lut_start_idx:metadata_idx]
            if lut_lines:
                lut_table = SEALCalibrationLoader._parse_lut_table(lut_lines)
        
        # Parse metadata
        metadata = SEALCalibrationLoader._parse_metadata(lines[metadata_idx])
        
        # Create stereo params (R, T are not stored in SEAL format)
        # We use identity/zero as placeholders
        stereo = StereoParams(
            K_left=left_params.K,
            dist_left=left_params.dist,
            K_right=right_params.K,
            dist_right=right_params.dist,
            R=np.eye(3),
            T=np.zeros((3, 1)),
            E=np.zeros((3, 3)),
            F=np.zeros((3, 3)),
            rms_error=0.0,
            img_size=resolution
        )
        
        return SEALCalibration(
            resolution=resolution,
            scale_factors=scale_factors,
            offset_center=offset_center,
            offset_tilt=offset_tilt,
            camera_left=left_params,
            camera_right=right_params,
            stereo=stereo,
            lut_table=lut_table,
            metadata=metadata
        )
    
    @staticmethod
    def _parse_camera_params(
        lines: list, 
        img_size: Tuple[int, int]
    ) -> CameraParams:
        """
        Parse camera intrinsic parameters from single line format
        
        Format: fx fy cx cy k1 k2 p1 p2 k3 [k4 k5 k6] [extra params...]
        The line can have 9-15 values, we use the first 9 for standard calibration
        """
        # Single line with all parameters
        vals = list(map(float, lines[0].split()))
        
        if len(vals) < 9:
            raise ValueError(f"Camera params line has {len(vals)} values, expected at least 9")
        
        # Extract standard parameters (first 9 values)
        fx, fy, cx, cy = vals[0:4]
        k1, k2, p1, p2, k3 = vals[4:9]
        
        # Optional additional distortion coefficients (k4, k5, k6)
        k4 = vals[9] if len(vals) > 9 else 0.0
        k5 = vals[10] if len(vals) > 10 else 0.0
        k6 = vals[11] if len(vals) > 11 else 0.0
        
        # Build intrinsic matrix
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Build distortion coefficients (8 coefficients for rational model)
        dist = np.array([[k1, k2, p1, p2, k3, k4, k5, k6]], dtype=np.float64)
        
        return CameraParams(
            K=K,
            dist=dist,
            img_size=img_size,
            rms_error=0.0
        )
    
    @staticmethod
    def _parse_lut_table(lines: list) -> Optional[np.ndarray]:
        """Parse LUT table from lines"""
        try:
            table_data = []
            for line in lines:
                if line and not line.startswith("***"):
                    values = list(map(float, line.split()))
                    if values:
                        table_data.append(values)
            
            if table_data:
                return np.array(table_data)
        except Exception:
            pass
        
        return None
    
    @staticmethod
    def _parse_metadata(line: str) -> Dict[str, str]:
        """Parse metadata line (***DevID:xxx***CalibrateDate:xxx***Type:xxx***)"""
        metadata = {}
        
        # Extract DevID
        dev_match = re.search(r"DevID:([^\*]+)", line)
        if dev_match:
            metadata['dev_id'] = dev_match.group(1)
        
        # Extract CalibrateDate
        date_match = re.search(r"CalibrateDate:([^\*]+)", line)
        if date_match:
            metadata['date'] = date_match.group(1)
        
        # Extract Type
        type_match = re.search(r"Type:([^\*]+)", line)
        if type_match:
            metadata['type'] = type_match.group(1)
        
        # Extract SoftVersion
        ver_match = re.search(r"SoftVersion:([^\s\*]+)", line)
        if ver_match:
            metadata['version'] = ver_match.group(1)
        
        return metadata
