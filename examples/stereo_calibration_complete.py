#!/usr/bin/env python3
"""
Complete Stereo Calibration Example with SEAL Format Export

This example provides full stereo calibration functionality including:
- Live camera capture or processing from existing images
- Multiple pattern types support
- SEAL format export with factory parameters preservation
- Detailed calibration report
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from datetime import datetime
import glob

from seal_calibration import (
    CameraCalibrator,
    StereoCalibrator,
    SEALCalibration,
    SEALCalibrationWriter,
    SEALCalibrationLoader,
    CameraParams,
    StereoParams,
)
from seal_calibration.pattern.chessboard import ChessboardDetector
from seal_calibration.pattern.charuco import CharucoDetector
from seal_calibration.pattern.circles import CirclesDetector


def capture_stereo_images(
    camera_left_idx, 
    camera_right_idx,
    detector,
    num_images,
    output_dir,
    auto_capture=True
):
    """Capture calibration images from stereo cameras"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap_left = cv2.VideoCapture(camera_left_idx)
    cap_right = cv2.VideoCapture(camera_right_idx)
    
    if not cap_left.isOpened() or not cap_right.isOpened():
        raise RuntimeError("Cannot open cameras")
    
    # Set resolution to 1280x720
    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    captured = 0
    auto_capture_delay = 0
    
    print(f"[INFO] Capturing {num_images} stereo image pairs...")
    if auto_capture:
        print("[INFO] Auto-capture mode: Pattern will be captured automatically")
    else:
        print("[INFO] Manual mode: Press SPACE to capture, Q to quit")
    
    while captured < num_images:
        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()
        
        if not ret_l or not ret_r:
            continue
        
        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
        
        # Detect patterns
        found_l, corners_l = detector.detect(gray_l)
        found_r, corners_r = detector.detect(gray_r)
        
        # Visualize
        disp_l = detector.draw_corners(frame_l.copy(), corners_l, found_l)
        disp_r = detector.draw_corners(frame_r.copy(), corners_r, found_r)
        
        # Draw info
        status_color = (0, 255, 0) if (found_l and found_r) else (0, 0, 255)
        cv2.putText(
            disp_l, f"Left - {captured}/{num_images}", 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, status_color, 2
        )
        cv2.putText(
            disp_r, f"Right - {captured}/{num_images}", 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, status_color, 2
        )
        
        # Stack images
        combined = np.hstack([disp_l, disp_r])
        cv2.imshow('Stereo Calibration', combined)
        
        # Auto-capture logic
        capture_now = False
        if auto_capture and found_l and found_r:
            auto_capture_delay += 1
            if auto_capture_delay > 30:  # ~1 second at 30fps
                capture_now = True
                auto_capture_delay = 0
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' ') and found_l and found_r:
            capture_now = True
        elif key == ord('q'):
            break
        
        # Capture image pair
        if capture_now:
            left_file = os.path.join(output_dir, f"{captured:03d}_left.png")
            right_file = os.path.join(output_dir, f"{captured:03d}_right.png")
            
            cv2.imwrite(left_file, frame_l)
            cv2.imwrite(right_file, frame_r)
            
            captured += 1
            print(f"[INFO] Captured {captured}/{num_images}")
    
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()
    
    return captured >= 4


def process_images_from_directory(images_dir, detector):
    """Process existing calibration images from directory"""
    
    print(f"[INFO] Processing images from: {images_dir}")
    
    # Find image pairs - support multiple naming patterns
    left_patterns = [
        os.path.join(images_dir, "*_left.png"),
        os.path.join(images_dir, "*_left.jpg"),
        os.path.join(images_dir, "left_*.png"),
        os.path.join(images_dir, "left_*.jpg"),
        os.path.join(images_dir, "gray_left_*.png"),
        os.path.join(images_dir, "gray_left_*.jpg"),
    ]
    
    right_patterns = [
        os.path.join(images_dir, "*_right.png"),
        os.path.join(images_dir, "*_right.jpg"),
        os.path.join(images_dir, "right_*.png"),
        os.path.join(images_dir, "right_*.jpg"),
        os.path.join(images_dir, "gray_right_*.png"),
        os.path.join(images_dir, "gray_right_*.jpg"),
    ]
    
    left_images = []
    right_images = []
    
    # Try each pattern until we find images
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
        print(f"[INFO] Processing pair {i+1}/{len(left_images)}...", end=' ')
        
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
    
    return objpoints, imgpoints_left, imgpoints_right, img_size


def print_calibration_report(camera_left, camera_right, stereo_params):
    """Print detailed calibration report"""
    
    print("\n" + "="*80)
    print(" STEREO CALIBRATION REPORT")
    print("="*80)
    
    # Overall metrics
    print(f"\n[CALIBRATION QUALITY]")
    print(f"  RMS Error: {stereo_params.rms_error:.4f} pixels")
    if stereo_params.rms_error < 0.5:
        print(f"  Quality: EXCELLENT ✓")
    elif stereo_params.rms_error < 1.0:
        print(f"  Quality: GOOD ✓")
    else:
        print(f"  Quality: NEEDS IMPROVEMENT ⚠")
    
    # Left camera
    print(f"\n[LEFT CAMERA - Laser/Front]")
    print(f"  Focal Length: fx={camera_left.fx:.2f}, fy={camera_left.fy:.2f}")
    print(f"  Principal Point: cx={camera_left.cx:.2f}, cy={camera_left.cy:.2f}")
    print(f"  Distortion: k1={camera_left.k1:.6f}, k2={camera_left.k2:.6f}, k3={camera_left.k3:.6f}")
    print(f"  Tangential: p1={camera_left.p1:.6f}, p2={camera_left.p2:.6f}")
    
    # Right camera
    print(f"\n[RIGHT CAMERA - UV/Tilted]")
    print(f"  Focal Length: fx={camera_right.fx:.2f}, fy={camera_right.fy:.2f}")
    print(f"  Principal Point: cx={camera_right.cx:.2f}, cy={camera_right.cy:.2f}")
    print(f"  Distortion: k1={camera_right.k1:.6f}, k2={camera_right.k2:.6f}, k3={camera_right.k3:.6f}")
    print(f"  Tangential: p1={camera_right.p1:.6f}, p2={camera_right.p2:.6f}")
    
    # Stereo parameters
    baseline = stereo_params.baseline
    print(f"\n[STEREO GEOMETRY]")
    print(f"  Baseline: {baseline:.2f} mm")
    print(f"  Translation: T = [{stereo_params.T[0,0]:.2f}, {stereo_params.T[1,0]:.2f}, {stereo_params.T[2,0]:.2f}] mm")
    
    # Rotation angles
    R = stereo_params.R
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    pitch = np.arctan2(-R[2,0], sy)
    yaw = np.arctan2(R[2,1], R[2,2])
    roll = np.arctan2(R[1,0], R[0,0])
    
    print(f"  Rotation (deg): pitch={np.degrees(pitch):.2f}°, yaw={np.degrees(yaw):.2f}°, roll={np.degrees(roll):.2f}°")
    
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Complete stereo calibration with SEAL format export"
    )
    
    # Camera configuration
    parser.add_argument("--left", type=int, default=0, 
                       help="Left camera index (default: 0)")
    parser.add_argument("--right", type=int, default=1, 
                       help="Right camera index (default: 1)")
    
    # Pattern configuration
    parser.add_argument("--pattern-type", type=str, default="chessboard",
                       choices=["chessboard", "charuco", "circles"],
                       help="Calibration pattern type")
    parser.add_argument("--rows", type=int, default=6,
                       help="Pattern rows (internal corners)")
    parser.add_argument("--cols", type=int, default=9,
                       help="Pattern columns (internal corners)")
    parser.add_argument("--square-size", type=float, default=25.0,
                       help="Square/circle size in mm")
    
    # Capture configuration
    parser.add_argument("--images", type=int, default=15,
                       help="Number of image pairs to capture")
    parser.add_argument("--no-auto-capture", action="store_true",
                       help="Disable auto-capture (use SPACE key)")
    parser.add_argument("--from-images", type=str, default=None,
                       help="Process existing images from directory")
    
    # Output configuration
    parser.add_argument("--output-dir", type=str, default="calib_imgs",
                       help="Directory for captured images")
    parser.add_argument("--template", type=str, default=None,
                       help="SEAL template file for factory parameters")
    parser.add_argument("--dev-id", type=str, default="JMS1006207",
                       help="Device ID for SEAL format")
    
    args = parser.parse_args()
    
    # Create pattern detector
    if args.pattern_type == "chessboard":
        detector = ChessboardDetector(args.rows, args.cols, args.square_size)
    elif args.pattern_type == "charuco":
        detector = CharucoDetector(args.rows, args.cols, args.square_size)
    elif args.pattern_type == "circles":
        detector = CirclesDetector(args.rows, args.cols, args.square_size)
    
    # Get calibration data
    if args.from_images:
        # Process existing images
        objpoints, imgpoints_left, imgpoints_right, img_size = \
            process_images_from_directory(args.from_images, detector)
    else:
        # Capture new images
        success = capture_stereo_images(
            args.left, args.right, detector, args.images, 
            args.output_dir, auto_capture=not args.no_auto_capture
        )
        
        if not success:
            print("[ERROR] Not enough images captured")
            return 1
        
        # Process captured images
        objpoints, imgpoints_left, imgpoints_right, img_size = \
            process_images_from_directory(args.output_dir, detector)
    
    if len(objpoints) < 4:
        print(f"[ERROR] Not enough valid images: {len(objpoints)}")
        return 1
    
    # Calibrate cameras
    print("\n[INFO] Calibrating left camera...")
    cam_calibrator = CameraCalibrator(detector)
    camera_left = cam_calibrator.calibrate(objpoints, imgpoints_left, img_size)
    
    print("[INFO] Calibrating right camera...")
    camera_right = cam_calibrator.calibrate(objpoints, imgpoints_right, img_size)
    
    print("[INFO] Performing stereo calibration...")
    stereo_calibrator = StereoCalibrator(detector)
    stereo_params = stereo_calibrator.calibrate(
        objpoints, imgpoints_left, imgpoints_right,
        camera_left, camera_right, img_size
    )
    
    # CRITICAL: Update camera parameters with stereo refined distortion coefficients
    # With CALIB_FIX_INTRINSIC | CALIB_RATIONAL_MODEL:
    # - Intrinsics (fx, fy, cx, cy) remain from individual calibration  
    # - Distortion coefficients are expanded from 5 to 8 and refined
    # We MUST use the refined 8-coefficient distortion from stereo_params
    camera_left = CameraParams(
        K=camera_left.K,  # Keep original K (CALIB_FIX_INTRINSIC)
        dist=stereo_params.dist_left,  # Use refined 8 coefficients
        img_size=img_size,
        rms_error=camera_left.rms_error
    )
    
    camera_right = CameraParams(
        K=camera_right.K,  # Keep original K (CALIB_FIX_INTRINSIC)
        dist=stereo_params.dist_right,  # Use refined 8 coefficients
        img_size=img_size,
        rms_error=camera_right.rms_error
    )
    
    # Print detailed report
    print_calibration_report(camera_left, camera_right, stereo_params)
    
    # Save NPZ format
    npz_file = "stereo_calibration.npz"
    np.savez(
        npz_file,
        K_left=stereo_params.K_left,
        dist_left=stereo_params.dist_left,
        K_right=stereo_params.K_right,
        dist_right=stereo_params.dist_right,
        R=stereo_params.R,
        T=stereo_params.T,
        E=stereo_params.E,
        F=stereo_params.F,
        img_size=img_size,
        rms_error=stereo_params.rms_error
    )
    print(f"[INFO] Calibration data saved to: {npz_file}")
    
    # Export to SEAL format if template provided
    if args.template:
        print(f"\n[INFO] Exporting to SEAL format...")
        
        try:
            # Load template
            template_calib = SEALCalibrationLoader.load(args.template)
            
            # Create SEAL calibration
            seal_calib = SEALCalibration(
                resolution=(1280, 720),
                scale_factors=template_calib.scale_factors,
                offset_center=template_calib.offset_center,
                offset_tilt=template_calib.offset_tilt,
                camera_left=camera_left,
                camera_right=camera_right,
                stereo=stereo_params,
                lut_table=template_calib.lut_table,
                metadata={
                    'dev_id': args.dev_id,
                    'date': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                    'type': template_calib.metadata.get('type', 'Factory-12'),
                    'version': '3.0.0.1116'
                }
            )
            
            # Write SEAL file
            seal_file = "stereo_calibration_seal.txt"
            SEALCalibrationWriter.write(seal_calib, seal_file, args.template)
            
            print(f"[INFO] SEAL format saved to: {seal_file}")
            print(f"\n[FACTORY PARAMETERS PRESERVED]")
            print(f"  Scale factors: {seal_calib.scale_factors}")
            print(f"  Offset center: {seal_calib.offset_center}")
            print(f"  Offset tilt: {seal_calib.offset_tilt}")
            
        except Exception as e:
            print(f"[WARNING] Could not export SEAL format: {e}")
    
    print(f"\n[SUCCESS] Calibration completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
