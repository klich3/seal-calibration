#!/usr/bin/env python3
"""
Basic camera calibration example

This example shows how to perform basic camera calibration
using the SEAL calibration library with a chessboard pattern.
"""

import cv2
import numpy as np
from pathlib import Path

from seal_calibration import CameraCalibrator, CameraParams
from seal_calibration.pattern.chessboard import ChessboardDetector


def main():
    """Run basic camera calibration"""
    
    # Configuration
    ROWS = 6  # Internal corners (rows)
    COLS = 9  # Internal corners (cols)
    SQUARE_SIZE = 3 # mm
    CAMERA_INDEX = 0
    NUM_IMAGES = 15
    
    # Create pattern detector
    detector = ChessboardDetector(ROWS, COLS, SQUARE_SIZE)
    
    # Create calibrator
    calibrator = CameraCalibrator(detector)
    
    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}")
        return
    
    # Get image size
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read from camera")
        return
    
    img_size = (frame.shape[1], frame.shape[0])
    print(f"[INFO] Image size: {img_size}")
    
    # Collect calibration images
    objpoints = []  # 3D points
    imgpoints = []  # 2D points
    
    captured = 0
    
    print(f"[INFO] Collecting {NUM_IMAGES} calibration images...")
    print("[INFO] Press SPACE to capture, Q to quit")
    
    while captured < NUM_IMAGES:
        ret, frame = cap.read()
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect pattern
        found, corners = detector.detect(gray)
        
        # Draw visualization
        display = frame.copy()
        if found:
            display = detector.draw_corners(display, corners, found)
        
        # Draw info
        cv2.putText(
            display, 
            f"Captured: {captured}/{NUM_IMAGES}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0) if found else (0, 0, 255), 
            2
        )
        
        cv2.imshow('Calibration', display)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' ') and found:
            # Capture image
            objpoints.append(detector.get_object_points())
            imgpoints.append(corners)
            captured += 1
            print(f"[INFO] Captured {captured}/{NUM_IMAGES}")
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if captured < 4:
        print("[ERROR] Not enough images for calibration")
        return
    
    # Perform calibration
    print("[INFO] Performing calibration...")
    camera_params = calibrator.calibrate(objpoints, imgpoints, img_size)
    
    # Print results
    print("\n=== Calibration Results ===")
    print(f"RMS Error: {camera_params.rms_error:.4f}")
    print(f"\nIntrinsic Matrix:")
    print(camera_params.K)
    print(f"\nDistortion Coefficients:")
    print(camera_params.dist)
    print(f"\nfx={camera_params.fx:.2f}, fy={camera_params.fy:.2f}")
    print(f"cx={camera_params.cx:.2f}, cy={camera_params.cy:.2f}")
    
    # Save results
    output_file = "camera_calibration.npz"
    np.savez(
        output_file,
        K=camera_params.K,
        dist=camera_params.dist,
        img_size=img_size,
        rms_error=camera_params.rms_error
    )
    print(f"\n[INFO] Results saved to {output_file}")


if __name__ == "__main__":
    main()
