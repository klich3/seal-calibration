"""SEAL calibration file writer"""
import re
from datetime import datetime
from typing import Optional, List
from ..models.seal_calib import SEALCalibration


class SEALCalibrationWriter:
    """Writes calibration to SEAL format"""
    
    FLOAT_FMT = "{:.6f}"
    
    @staticmethod
    def write(
        calib: SEALCalibration, 
        output_path: str, 
        template_path: Optional[str] = None
    ):
        """
        Write calibration to SEAL format
        
        Args:
            calib: SEALCalibration object
            output_path: Output file path
            template_path: Template file to preserve LUT/Gray code data
        """
        if template_path:
            SEALCalibrationWriter._write_with_template(
                calib, output_path, template_path
            )
        else:
            SEALCalibrationWriter._write_direct(calib, output_path)
    
    @staticmethod
    def _write_direct(calib: SEALCalibration, output_path: str):
        """Write calibration directly using to_seal_format()"""
        content = calib.to_seal_format()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
            f.write('\n')
    
    @staticmethod
    def _write_with_template(
        calib: SEALCalibration, 
        output_path: str, 
        template_path: str
    ):
        """
        Write calibration preserving template structure and LUT/Gray code data
        
        CRITICAL: Lines 2, 3, 4 are FACTORY PARAMETERS and MUST NOT be modified
        """
        with open(template_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.read().splitlines()
        
        if len(lines) < 10:
            raise ValueError("Template file too short")
        
        # Line 1: Resolution (ALWAYS update to 1280x720)
        lines[0] = f"{calib.resolution[0]} {calib.resolution[1]}"
        
        # Lines 2-4: FACTORY PARAMETERS - DO NOT MODIFY
        # These are preserved from template automatically
        
        # Line 5: Left camera intrinsics (single line format: 15 values)
        lines[4] = SEALCalibrationWriter._format_camera_line(calib.camera_left, lines[4])
        
        # Line 6: Right camera intrinsics - PRESERVE FROM TEMPLATE
        # Historical behavior: only update left camera, keep right camera from template
        # This is because the right camera (UV) calibration is more stable and factory-calibrated
        # lines[5] = SEALCalibrationWriter._format_camera_line(calib.camera_right, lines[5])
        # Lines 2-4 are factory parameters, line 6 is also preserved from factory
        
        # Line 7: Projector parameters - preserve from template
        # (We don't modify projector calibration)
        
        # Find metadata line
        metadata_idx = len(lines) - 1
        for i in range(len(lines) - 1, 5, -1):  # Start search after cameras
            if lines[i].startswith("***DevID:"):
                metadata_idx = i
                break
        
        # Update metadata
        lines[metadata_idx] = SEALCalibrationWriter._update_metadata_line(
            lines[metadata_idx],
            calib.metadata.get('dev_id'),
            calib.metadata.get('version')
        )
        
        # Write output
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
            f.write('\n')
    
    @staticmethod
    def _format_camera_line(camera_params, template_line: str) -> str:
        """
        Format camera parameters in single line format (15 values)
        Preserves extra parameters from template (values 10-15)
        
        Format: fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6 [extra_1 extra_2 extra_3]
        """
        fmt = SEALCalibrationWriter.FLOAT_FMT
        
        # Extract values 10-15 from template if they exist
        template_vals = template_line.split()
        extra_params = template_vals[9:15] if len(template_vals) >= 15 else []
        
        # Build new line with updated calibration values (first 9)
        parts = [
            fmt.format(camera_params.fx),
            fmt.format(camera_params.fy),
            fmt.format(camera_params.cx),
            fmt.format(camera_params.cy),
            fmt.format(camera_params.k1),
            fmt.format(camera_params.k2),
            fmt.format(camera_params.p1),
            fmt.format(camera_params.p2),
            fmt.format(camera_params.k3),
        ]
        
        # Add k4, k5, k6 (values 10-12)
        parts.extend([
            fmt.format(camera_params.k4),
            fmt.format(camera_params.k5),
            fmt.format(camera_params.k6),
        ])
        
        # Preserve extra parameters from template (values 13-15)
        if len(extra_params) > 3:
            parts.extend(extra_params[3:6])  # Add extra params 13-15
        
        return ' '.join(parts)
    
    @staticmethod
    def _format_intrinsic_line(fx: float, fy: float, cx: float, cy: float) -> str:
        """Format intrinsic parameters line (DEPRECATED - kept for compatibility)"""
        fmt = SEALCalibrationWriter.FLOAT_FMT
        return f"{fmt.format(fx)} {fmt.format(fy)} {fmt.format(cx)} {fmt.format(cy)}"
    
    @staticmethod
    def _format_distortion_line(k1: float, k2: float, k3: float, k4: float) -> str:
        """Format distortion parameters line"""
        fmt = SEALCalibrationWriter.FLOAT_FMT
        return f"{fmt.format(k1)} {fmt.format(k2)} {fmt.format(k3)} {fmt.format(k4)}"
    
    @staticmethod
    def _update_metadata_line(
        old_line: str, 
        dev_id: Optional[str], 
        soft_version: Optional[str]
    ) -> str:
        """
        Update metadata line preserving Type from template
        Updates DevID and CalibrateDate
        Sets Type to 'Sychev-calibration' for recalibrated files
        """
        # Extract Type from template (but override to Sychev-calibration)
        type_val = "Sychev-calibration"  # Always use Sychev-calibration for recalibration
        
        # Extract SoftVersion if not provided
        if soft_version is None:
            sv_match = re.search(r"SoftVersion:([^\s\*]+)", old_line)
            soft_version = sv_match.group(1) if sv_match else "3.0.0.1116"
        
        # Extract DevID if not provided
        if dev_id is None:
            dev_match = re.search(r"DevID:([^\*]+)", old_line)
            dev_id = dev_match.group(1) if dev_match else "UNKNOWN"
        
        # Generate new timestamp
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        return (
            f"***DevID:{dev_id}***CalibrateDate:{now}***"
            f"Type:{type_val}***SoftVersion:{soft_version}"
        )
