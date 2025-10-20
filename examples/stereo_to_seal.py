#!/usr/bin/env python3
"""
Export stereo calibration to SEAL format

This script performs stereo calibration and exports results
to SEAL calibration file format, compatible with historical methods.
"""

import cv2
import numpy as np
import argparse
import glob
import os
from datetime import datetime

from seal_calibration import CameraCalibrator, StereoCalibrator
from seal_calibration.pattern.chessboard import ChessboardDetector
from seal_calibration.io import SEALCalibrationWriter, SEALCalibrationLoader
from seal_calibration.models import SEALCalibration, StereoParams
from seal_calibration.utils import print_calibration_summary


def process_from_images(images_dir, detector):
    """Process calibration images from directory"""
    
    print(f"[INFO] Processing images from: {images_dir}")
    
    # Find image pairs - support multiple naming patterns
    left_patterns = [
        os.path.join(images_dir, "*_left.png"),
        os.path.join(images_dir, "*_left.jpg"),
        os.path.join(images_dir, "left_*.png"),
        os.path.join(images_dir, "left_*.jpg"),
    ]
    
    right_patterns = [
        os.path.join(images_dir, "*_right.png"),
        os.path.join(images_dir, "*_right.jpg"),
        os.path.join(images_dir, "right_*.png"),
        os.path.join(images_dir, "right_*.jpg"),
    ]
    
    left_images = []
    right_images = []
    
    for pattern in left_patterns:
        left_images = sorted(glob.glob(pattern))
        if left_images:
            print(f"[INFO] Found left images with pattern: {os.path.basename(pattern)}")
            break
    
    for pattern in right_patterns:
        right_images = sorted(glob.glob(pattern))
        if right_images:
            print(f"[INFO] Found right images with pattern: {os.path.basename(pattern)}")
            break
    
    if not left_images or not right_images:
        raise RuntimeError(f"No image pairs found in {images_dir}")
    
    if len(left_images) != len(right_images):
        raise RuntimeError(
            f"Mismatched image counts: {len(left_images)} left vs {len(right_images)} right"
        )
    
    print(f"[INFO] Found {len(left_images)} image pairs")
    
    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    img_size = None
    
    for i, (left_file, right_file) in enumerate(zip(left_images, right_images)):
        left_name = os.path.basename(left_file)
        right_name = os.path.basename(right_file)
        print(f"[INFO] Processing pair {i+1}/{len(left_images)} ({left_name}, {right_name})...", end=' ')
        
        img_l = cv2.imread(left_file)
        img_r = cv2.imread(right_file)
        
        if img_size is None:
            img_size = (img_l.shape[1], img_l.shape[0])
        
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        
        found_l, corners_l = detector.detect(gray_l)
        found_r, corners_r = detector.detect(gray_r)
        
        if found_l and found_r:
            objpoints.append(detector.get_object_points())
            imgpoints_left.append(corners_l)
            imgpoints_right.append(corners_r)
            print("✓")
        else:
            print("✗ Pattern not found")
    
    print(f"[INFO] Successfully processed {len(objpoints)} pairs")
    
    if len(objpoints) < 4:
        raise RuntimeError(f"Not enough valid images: {len(objpoints)}")
    
    return objpoints, imgpoints_left, imgpoints_right, img_size


def main():
    """Run stereo calibration and export to SEAL format"""
    
    parser = argparse.ArgumentParser(description="Export stereo calibration to SEAL format")
    parser.add_argument("--from-images", type=str, required=True,
                       help="Process images from directory")
    parser.add_argument("--template", type=str, required=True,
                       help="SEAL template file to preserve factory parameters")
    parser.add_argument("--output", type=str, default="stereo_calibration_seal.txt",
                       help="Output SEAL calibration file")
    parser.add_argument("--rows", type=int, default=6, help="Pattern rows")
    parser.add_argument("--cols", type=int, default=9, help="Pattern columns")
    parser.add_argument("--square-size", type=float, default=25.0, help="Square size (mm)")
    parser.add_argument("--dev-id", type=str, default="JMS1006207", help="Device ID")
    args = parser.parse_args()
    
    # Configuration
    ROWS = args.rows
    COLS = args.cols
    SQUARE_SIZE = args.square_size
    
    # Create detector
    detector = ChessboardDetector(ROWS, COLS, SQUARE_SIZE)
    
    # Process images
    objpoints, imgpoints_left, imgpoints_right, img_size = \
        process_from_images(args.from_images, detector)
    
    # Calibrate individual cameras
    print(f"[INFO] Calibrando con {len(objpoints)} pares...")
    print("[INFO] Calibrating left camera...")
    cam_calibrator = CameraCalibrator(detector)
    camera_left = cam_calibrator.calibrate(objpoints, imgpoints_left, img_size)
    
    print("[INFO] Calibrating right camera...")
    camera_right = cam_calibrator.calibrate(objpoints, imgpoints_right, img_size)
    
    # Stereo calibration
    print("[INFO] Calibración estéreo...")
    stereo_calibrator = StereoCalibrator(detector)
    stereo_params = stereo_calibrator.calibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        camera_left,
        camera_right,
        img_size
    )
    
    # Print detailed summary
    print_calibration_summary(
        stereo_params.K_left,
        stereo_params.K_right,
        stereo_params.dist_left,
        stereo_params.dist_right,
        stereo_params.R,
        stereo_params.T,
        stereo_params.rms_error,
        img_size
    )
    
    print("[INFO] Calibración completada")
    
    # Load template to preserve factory parameters
    print(f"\n[INFO] Loading template: {args.template}")
    template_calib = SEALCalibrationLoader.load(args.template)
    
    # Create new SEAL calibration with updated camera parameters
    seal_calib = SEALCalibration(
        resolution=(1280, 720),  # Force SEAL resolution
        scale_factors=template_calib.scale_factors,  # Preserve from template
        offset_center=template_calib.offset_center,  # Preserve from template
        offset_tilt=template_calib.offset_tilt,  # Preserve from template
        camera_left=camera_left,
        camera_right=camera_right,
        stereo=stereo_params,
        lut_table=template_calib.lut_table,  # Preserve from template
        metadata={
            'dev_id': args.dev_id,
            'date': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            'type': 'Sychev-calibration',
            'version': '3.0.0.1116'
        }
    )
    
    # Write to SEAL format
    print(f"\n[INFO] Writing SEAL calibration file...")
    SEALCalibrationWriter.write(seal_calib, args.output, args.template)
    
    print(f"[INFO] SEAL calibration file generated: {args.output}")
    print(f"  NOTA: Líneas 2, 3, 4 preservadas de la plantilla (parámetros de fábrica)")
    print(f"  Solo se actualizaron: intrínsecos, distorsión y metadatos")


if __name__ == "__main__":
    main()
