"""Visualization utilities for calibration"""
import cv2
import numpy as np
from typing import List, Optional, Tuple


def draw_calibration_info(
    image: np.ndarray,
    camera_name: str,
    frame_count: int,
    fps: float,
    pattern_found: bool,
    total_captures: int,
    target_captures: int
) -> np.ndarray:
    """
    Draw calibration information overlay on image
    
    Args:
        image: Input image
        camera_name: Camera name
        frame_count: Current frame count
        fps: Frames per second
        pattern_found: Whether pattern is currently detected
        total_captures: Number of captured calibration images
        target_captures: Target number of calibration images
        
    Returns:
        Image with overlay
    """
    overlay = image.copy()
    
    # Draw semi-transparent background
    cv2.rectangle(overlay, (0, 0), (400, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    
    # Draw text
    y_offset = 30
    cv2.putText(
        image, f"Camera: {camera_name}", 
        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, (255, 255, 255), 2
    )
    
    y_offset += 30
    cv2.putText(
        image, f"Frame: {frame_count} | FPS: {fps:.1f}", 
        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
        0.6, (255, 255, 255), 1
    )
    
    y_offset += 30
    status_color = (0, 255, 0) if pattern_found else (0, 0, 255)
    status_text = "DETECTED" if pattern_found else "NOT FOUND"
    cv2.putText(
        image, f"Pattern: {status_text}", 
        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
        0.6, status_color, 2
    )
    
    y_offset += 30
    progress_color = (0, 255, 0) if total_captures >= target_captures else (0, 255, 255)
    cv2.putText(
        image, f"Captures: {total_captures}/{target_captures}", 
        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
        0.6, progress_color, 2
    )
    
    return image


def draw_epipolar_lines(
    img_left: np.ndarray,
    img_right: np.ndarray,
    points_left: np.ndarray,
    F: np.ndarray,
    num_lines: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw epipolar lines for stereo verification
    
    Args:
        img_left: Left image
        img_right: Right image
        points_left: Points in left image
        F: Fundamental matrix
        num_lines: Number of lines to draw
        
    Returns:
        Tuple of (left image with lines, right image with lines)
    """
    h_left, w_left = img_left.shape[:2]
    h_right, w_right = img_right.shape[:2]
    
    # Select random points
    indices = np.random.choice(
        len(points_left), 
        min(num_lines, len(points_left)), 
        replace=False
    )
    
    img_left_lines = img_left.copy()
    img_right_lines = img_right.copy()
    
    for idx in indices:
        pt = points_left[idx]
        
        # Compute epipolar line in right image
        line = F @ np.array([pt[0], pt[1], 1])
        
        # Draw point in left image
        cv2.circle(
            img_left_lines, 
            (int(pt[0]), int(pt[1])), 
            5, (0, 255, 0), -1
        )
        
        # Draw line in right image
        x0, y0 = 0, int(-line[2] / line[1])
        x1, y1 = w_right, int(-(line[2] + line[0] * w_right) / line[1])
        
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(img_right_lines, (x0, y0), (x1, y1), color, 2)
    
    return img_left_lines, img_right_lines


def visualize_reprojection_error(
    image: np.ndarray,
    detected_points: np.ndarray,
    projected_points: np.ndarray
) -> np.ndarray:
    """
    Visualize reprojection errors
    
    Args:
        image: Input image
        detected_points: Detected points
        projected_points: Reprojected points
        
    Returns:
        Image with error visualization
    """
    vis = image.copy()
    
    for det, proj in zip(detected_points, projected_points):
        det_pt = tuple(det[0].astype(int))
        proj_pt = tuple(proj[0].astype(int))
        
        # Draw detected point (green)
        cv2.circle(vis, det_pt, 3, (0, 255, 0), -1)
        
        # Draw projected point (red)
        cv2.circle(vis, proj_pt, 3, (0, 0, 255), -1)
        
        # Draw error line
        cv2.line(vis, det_pt, proj_pt, (255, 0, 0), 1)
    
    return vis


def create_calibration_grid_visualization(
    image: np.ndarray,
    points: np.ndarray,
    img_size: Tuple[int, int],
    grid_size: Tuple[int, int] = (4, 3)
) -> np.ndarray:
    """
    Visualize calibration point coverage
    
    Args:
        image: Input image
        points: Calibration points
        img_size: Image size (width, height)
        grid_size: Grid divisions
        
    Returns:
        Image with coverage visualization
    """
    vis = image.copy()
    width, height = img_size
    grid_w, grid_h = grid_size
    
    cell_w = width / grid_w
    cell_h = height / grid_h
    
    # Draw grid
    for i in range(1, grid_w):
        x = int(i * cell_w)
        cv2.line(vis, (x, 0), (x, height), (100, 100, 100), 1)
    
    for i in range(1, grid_h):
        y = int(i * cell_h)
        cv2.line(vis, (0, y), (width, y), (100, 100, 100), 1)
    
    # Mark covered cells
    covered = np.zeros((grid_h, grid_w), dtype=bool)
    
    for pt in points:
        x, y = int(pt[0, 0]), int(pt[0, 1])
        cell_x = min(int(x / cell_w), grid_w - 1)
        cell_y = min(int(y / cell_h), grid_h - 1)
        covered[cell_y, cell_x] = True
    
    # Draw coverage overlay
    overlay = vis.copy()
    for i in range(grid_h):
        for j in range(grid_w):
            x1 = int(j * cell_w)
            y1 = int(i * cell_h)
            x2 = int((j + 1) * cell_w)
            y2 = int((i + 1) * cell_h)
            
            color = (0, 255, 0) if covered[i, j] else (0, 0, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    
    cv2.addWeighted(overlay, 0.2, vis, 0.8, 0, vis)
    
    return vis
