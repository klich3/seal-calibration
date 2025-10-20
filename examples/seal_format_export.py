#!/usr/bin/env python3
"""
SEAL format export example

This example shows how to export calibration results
to SEAL scanner format.
"""

import numpy as np
from datetime import datetime

from seal_calibration import (
    SEALCalibration,
    SEALCalibrationWriter,
    SEALCalibrationLoader,
    CameraParams,
    StereoParams,
)


def main():
    """Export calibration to SEAL format"""
    
    # Load stereo calibration results
    calib_file = "stereo_calibration.npz"
    
    try:
        data = np.load(calib_file)
    except FileNotFoundError:
        print(f"[ERROR] Calibration file not found: {calib_file}")
        print("[INFO] Run stereo_calibration.py first")
        return
    
    # Create camera parameters
    img_size = tuple(data['img_size'])
    
    camera_left = CameraParams(
        K=data['K_left'],
        dist=data['dist_left'],
        img_size=img_size,
        rms_error=float(data['rms_error'])
    )
    
    camera_right = CameraParams(
        K=data['K_right'],
        dist=data['dist_right'],
        img_size=img_size,
        rms_error=float(data['rms_error'])
    )
    
    # Create stereo parameters
    stereo = StereoParams(
        K_left=data['K_left'],
        dist_left=data['dist_left'],
        K_right=data['K_right'],
        dist_right=data['dist_right'],
        R=data['R'],
        T=data['T'],
        E=np.zeros((3, 3)),
        F=np.zeros((3, 3)),
        rms_error=float(data['rms_error']),
        img_size=img_size
    )
    
    # Create SEAL calibration object
    # NOTE: For a real SEAL scanner, you should load a template file
    # to preserve factory-calibrated parameters (lines 2-4)
    # For this example, we use estimated default values
    
    template_file = "calibJMS1006207.txt"
    
    # Try to load template if available
    try:
        template_calib = SEALCalibrationLoader.load(template_file)
        scale_factors = template_calib.scale_factors
        offset_center = template_calib.offset_center
        offset_tilt = template_calib.offset_tilt
        lut_table = template_calib.lut_table
        print(f"[INFO] Loaded factory parameters from template: {template_file}")
    except (FileNotFoundError, Exception) as e:
        print(f"[WARNING] Could not load template: {e}")
        print("[INFO] Using default estimated values (NOT for production use!)")
        # Default estimated values - NOT ACCURATE FOR REAL SCANNING
        scale_factors = (11.6, 4.4)  # baseline_scale, depth_scale
        offset_center = (162, 110)   # dx, dy
        offset_tilt = (4, -80)       # tilt, height
        lut_table = None
    
    seal_calib = SEALCalibration(
        resolution=(1280, 720),  # Always 1280x720
        scale_factors=scale_factors,
        offset_center=offset_center,
        offset_tilt=offset_tilt,
        camera_left=camera_left,
        camera_right=camera_right,
        stereo=stereo,
        lut_table=lut_table,
        metadata={
            'dev_id': 'JMS1006207',
            'date': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            'type': 'Sychev-calibration',
            'version': '3.0.0.1116'
        }
    )
    
    # Write SEAL format file
    output_file = "stereo_calibration_seal.txt"
    
    # Only use template if we successfully loaded it
    template_path = template_file if 'template_calib' in locals() and template_calib is not None else None
    
    SEALCalibrationWriter.write(
        seal_calib,
        output_file,
        template_path=template_path
    )
    
    print(f"\n[INFO] SEAL calibration file exported to: {output_file}")
    
    if 'template_calib' in locals() and template_calib is not None:
        print("\n=== CRITICAL INFORMATION ===")
        print("Lines 2-4 contain FACTORY PARAMETERS and were NOT modified:")
        print(f"  Line 2 (Scale factors): {seal_calib.scale_factors}")
        print(f"  Line 3 (Offset center): {seal_calib.offset_center}")
        print(f"  Line 4 (Offset tilt): {seal_calib.offset_tilt}")
        print("\nOnly camera intrinsic parameters (lines 5-10) were updated.")
    else:
        print("\n⚠️  WARNING ⚠️")
        print("This calibration uses DEFAULT estimated values, NOT factory parameters!")
        print("For production use, you MUST provide a factory calibration template file.")
        print("The output file is for TESTING purposes only.")
    
    print("\nRecommended scanner_settings_Seal.ini tolerances:")
    print("  BaseTol=20")
    print("  TolDown=0")
    print("  TolUp=50")
    print("  TolRadius=5")


if __name__ == "__main__":
    main()
