"""Basic scanning example"""
from seal_scanner import SEALScanner
import cv2

# Initialize scanner
scanner = SEALScanner(calibration_file="stereo_calibration_seal.txt")

# Start scan
scanner.start_scan()

# Capture and process frames
cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)

for i in range(100):
    ret_l, img_l = cap_left.read()
    ret_r, img_r = cap_right.read()
    
    if ret_l and ret_r:
        pcd = scanner.process_frame(img_l, img_r)
        if pcd:
            print(f"Frame {i}: OK")
        else:
            print(f"Frame {i}: No pointcloud")

# Finalize
scanner.finalize_scan("output.ply")

cap_left.release()
cap_right.release()
