"""SEAL calibration complete model"""
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional
from .camera_params import CameraParams
from .stereo_params import StereoParams


@dataclass
class SEALCalibration:
    """Complete SEAL calibration parameters
    
    This class represents the complete calibration for a SEAL 3D scanner,
    including camera intrinsics, stereo parameters, projector calibration,
    and factory-calibrated parameters.
    
    Attributes:
        resolution: Image resolution (width, height)
        scale_factors: Scale factors (baseline_scale, depth_scale)
        offset_center: Projection center offset (dx, dy)
        offset_tilt: Projector tilt offset (tilt, height)
        camera_left: Left camera parameters
        camera_right: Right camera parameters
        stereo: Stereo calibration parameters
        lut_table: Look-up table for Gray code (optional)
        metadata: Device metadata (DevID, date, type, version)
    """
    resolution: Tuple[int, int]
    scale_factors: Tuple[float, float]
    offset_center: Tuple[int, int]
    offset_tilt: Tuple[int, int]
    camera_left: CameraParams
    camera_right: CameraParams
    stereo: StereoParams
    lut_table: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)
    
    def to_seal_format(self) -> str:
        """Convert to SEAL text format"""
        lines = []
        
        # Line 1: Resolution
        lines.append(f"{self.resolution[0]} {self.resolution[1]}")
        
        # Line 2: Scale factors (FACTORY - DO NOT MODIFY)
        lines.append(f"{self.scale_factors[0]:.6f} {self.scale_factors[1]:.6f}")
        
        # Line 3: Offset center (FACTORY - DO NOT MODIFY)
        lines.append(f"{self.offset_center[0]} {self.offset_center[1]}")
        
        # Line 4: Offset tilt (FACTORY - DO NOT MODIFY)
        lines.append(f"{self.offset_tilt[0]} {self.offset_tilt[1]}")
        
        # Lines 5-6: Camera intrinsics (single line format: 15 values)
        # Format: fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6 extra1 extra2 extra3
        
        # Left camera (line 5)
        lines.append(f"{self.camera_left.fx:.6f} {self.camera_left.fy:.6f} "
                    f"{self.camera_left.cx:.6f} {self.camera_left.cy:.6f} "
                    f"{self.camera_left.k1:.6f} {self.camera_left.k2:.6f} "
                    f"{self.camera_left.p1:.6f} {self.camera_left.p2:.6f} "
                    f"{self.camera_left.k3:.6f} {self.camera_left.k4:.6f} "
                    f"{self.camera_left.k5:.6f} {self.camera_left.k6:.6f} "
                    f"0.000000 0.000000 0.000000")  # Placeholder extra params
        
        # Right camera (line 6)
        lines.append(f"{self.camera_right.fx:.6f} {self.camera_right.fy:.6f} "
                    f"{self.camera_right.cx:.6f} {self.camera_right.cy:.6f} "
                    f"{self.camera_right.k1:.6f} {self.camera_right.k2:.6f} "
                    f"{self.camera_right.p1:.6f} {self.camera_right.p2:.6f} "
                    f"{self.camera_right.k3:.6f} {self.camera_right.k4:.6f} "
                    f"{self.camera_right.k5:.6f} {self.camera_right.k6:.6f} "
                    f"0.000000 0.000000 0.000000")  # Placeholder extra params
        
        # Line 7: Projector parameters (placeholder - should use Gray Code calibration)
        lines.append(f"2800.000000 2477.000000 542.000000 353.000000 "
                    f"-0.220000 -0.295000 -0.000011 -0.000875 "
                    f"4.029662 -0.007206 -0.133619 -0.003511 "
                    f"28.377724 0.011272 1.662601")  # Placeholder projector params
        
        # LUT table if present
        if self.lut_table is not None:
            for row in self.lut_table:
                lines.append(" ".join(map(str, row)))
        
        # Metadata footer
        if self.metadata:
            dev_id = self.metadata.get('dev_id', 'UNKNOWN')
            date = self.metadata.get('date', '2024-01-01_00-00-00')
            calib_type = 'Sychev-calibration'  # Always use Sychev-calibration for recalibrated files
            version = self.metadata.get('version', '3.0.0.1116')
            
            footer = f"***DevID:{dev_id}***CalibrateDate:{date}***Type:{calib_type}***SoftVersion:{version}"
            lines.append(footer)
        
        return "\n".join(lines)
