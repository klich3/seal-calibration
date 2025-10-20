#!/usr/bin/env python3
"""
Stereo camera calibration example

This example shows how to perform stereo calibration
using the SEAL calibration library, with support for both
live camera capture and processing existing images.
"""

import cv2
import numpy as np
import argparse
import glob
import os
from pathlib import Path

from seal_calibration import CameraCalibrator, StereoCalibrator
from seal_calibration.pattern.chessboard import ChessboardDetector
from seal_calibration.utils import (
    calculate_reprojection_errors,
    validate_calibration,
    print_calibration_summary
)


def process_from_images(images_dir, detector):
    """Process calibration images from directory"""
    
    print(f"[INFO] Processing images from: {images_dir}")
    
    # Find image pairs - support multiple naming patterns
    # Patterns: *_left.*, left_*.*, gray_left_*.*
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
    
    if len(objpoints) < 4:
        raise RuntimeError(f"Not enough valid images: {len(objpoints)}")
    
    return objpoints, imgpoints_left, imgpoints_right, img_size


def capture_from_cameras(camera_left, camera_right, detector, num_images):
    """Capture calibration images from live cameras"""
    
    # Open cameras
    cap_left = cv2.VideoCapture(camera_left)
    cap_right = cv2.VideoCapture(camera_right)
    
    if not cap_left.isOpened() or not cap_right.isOpened():
        raise RuntimeError("Cannot open cameras")
    
    # Set resolution to 1280x720
    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Get image size
    ret_l, frame_l = cap_left.read()
    ret_r, frame_r = cap_right.read()
    
    if not ret_l or not ret_r:
        cap_left.release()
        cap_right.release()
        raise RuntimeError("Cannot read from cameras")
    
    img_size = (frame_l.shape[1], frame_l.shape[0])
    print(f"[INFO] Image size: {img_size}")
    
    # Collect calibration images
    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    
    captured = 0
    
    print(f"[INFO] Collecting {num_images} stereo calibration images...")
    print("[INFO] Press SPACE to capture, Q to quit")
    
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
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' ') and found_l and found_r:
            objpoints.append(detector.get_object_points())
            imgpoints_left.append(corners_l)
            imgpoints_right.append(corners_r)
            captured += 1
            print(f"[INFO] Captured {captured}/{num_images}")
        
        elif key == ord('q'):
            break
    
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()
    
    if captured < 4:
        raise RuntimeError("Not enough images for calibration")
    
    return objpoints, imgpoints_left, imgpoints_right, img_size


def main():
    """Run stereo camera calibration"""
    
    parser = argparse.ArgumentParser(description="Stereo camera calibration")
    parser.add_argument("--left", type=int, default=0, help="Left camera index")
    parser.add_argument("--right", type=int, default=1, help="Right camera index")
    parser.add_argument("--rows", type=int, default=6, help="Pattern rows")
    parser.add_argument("--cols", type=int, default=9, help="Pattern columns")
    parser.add_argument("--square-size", type=float, default=25.0, help="Square size (mm)")
    parser.add_argument("--images", type=int, default=15, help="Number of images to capture")
    parser.add_argument("--from-images", type=str, default=None,
                       help="Process existing images from directory")
    args = parser.parse_args()
    
    # Configuration
    ROWS = args.rows
    COLS = args.cols
    SQUARE_SIZE = args.square_size
    CAMERA_LEFT = args.left
    CAMERA_RIGHT = args.right
    NUM_IMAGES = args.images
    
    # Create detector
    detector = ChessboardDetector(ROWS, COLS, SQUARE_SIZE)
    
    # Get calibration data
    if args.from_images:
        # Process existing images
        objpoints, imgpoints_left, imgpoints_right, img_size = \
            process_from_images(args.from_images, detector)
    else:
        # Capture from live cameras
        objpoints, imgpoints_left, imgpoints_right, img_size = \
            capture_from_cameras(CAMERA_LEFT, CAMERA_RIGHT, detector, NUM_IMAGES)
    
    # Calibrate individual cameras
    print("[INFO] Calibrating left camera...")
    cam_calibrator = CameraCalibrator(detector)
    camera_left = cam_calibrator.calibrate(objpoints, imgpoints_left, img_size)
    
    print("[INFO] Calibrating right camera...")
    camera_right = cam_calibrator.calibrate(objpoints, imgpoints_right, img_size)
    
    # Stereo calibration
    print("[INFO] Performing stereo calibration...")
    stereo_calibrator = StereoCalibrator(detector)
    stereo_params = stereo_calibrator.calibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        camera_left,
        camera_right,
        img_size
    )
    
    # Print results
    print("\n" + "="*60)
    print(" STEREO CALIBRATION RESULTS")
    print("="*60)
    print(f"\nRMS Error: {stereo_params.rms_error:.4f} pixels")
    
    # Quality assessment
    if stereo_params.rms_error < 0.5:
        print(f"Quality: EXCELLENT ✓")
    elif stereo_params.rms_error < 1.0:
        print(f"Quality: GOOD ✓")
    else:
        print(f"Quality: NEEDS IMPROVEMENT ⚠")
        print(f"  Recommendation: Capture more images with better lighting")
    
    print(f"\nBaseline: {stereo_params.baseline:.2f} mm")
    
    # Verify resolution
    if img_size == (1280, 720):
        print(f"Resolution: {img_size[0]}x{img_size[1]} ✓")
    else:
        print(f"Resolution: {img_size[0]}x{img_size[1]} ⚠ (Recommended: 1280x720)")
    
    print(f"\nRotation Matrix:")
    print(stereo_params.R)
    print(f"\nTranslation Vector:")
    print(stereo_params.T.T)
    print("\n" + "="*60)
    
    # Save results
    output_file = "stereo_calibration.npz"
    np.savez(
        output_file,
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
    print(f"\n[INFO] Results saved to {output_file}")


if __name__ == "__main__":
    main()
