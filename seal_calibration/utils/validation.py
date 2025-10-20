"""Calibration validation and reporting utilities"""
import cv2
import numpy as np
from typing import List, Tuple


def calculate_reprojection_errors(
    objpoints: List[np.ndarray],
    imgpoints_left: List[np.ndarray],
    imgpoints_right: List[np.ndarray],
    K_left: np.ndarray,
    dist_left: np.ndarray,
    K_right: np.ndarray,
    dist_right: np.ndarray,
    R: np.ndarray,
    T: np.ndarray
) -> Tuple[List[float], List[float]]:
    """Calculate reprojection errors for each image"""
    errors_left = []
    errors_right = []
    
    rvec, _ = cv2.Rodrigues(R)
    
    for i, objp in enumerate(objpoints):
        # Project 3D points to 2D for left camera
        imgpoints_left_proj, _ = cv2.projectPoints(
            objp, np.zeros((3,1)), np.zeros((3,1)), K_left, dist_left)
        # Reshape to match input format
        imgpoints_left_proj = imgpoints_left_proj.reshape(-1, 2)
        imgpoints_left_i = imgpoints_left[i].reshape(-1, 2)
        error_left = np.sqrt(np.sum((imgpoints_left_i - imgpoints_left_proj)**2)) / len(imgpoints_left_proj)
        errors_left.append(error_left)
        
        # For right camera, use R and T
        imgpoints_right_proj, _ = cv2.projectPoints(
            objp, rvec, T, K_right, dist_right)
        # Reshape to match input format
        imgpoints_right_proj = imgpoints_right_proj.reshape(-1, 2)
        imgpoints_right_i = imgpoints_right[i].reshape(-1, 2)
        error_right = np.sqrt(np.sum((imgpoints_right_i - imgpoints_right_proj)**2)) / len(imgpoints_right_proj)
        errors_right.append(error_right)
    
    return errors_left, errors_right


def validate_calibration(
    ret: float,
    F: np.ndarray,
    img_size: Tuple[int, int],
    errors_left: List[float],
    errors_right: List[float]
):
    """Validate calibration quality"""
    print(f"\n[VALIDACIÓN DE CALIBRACIÓN]")
    
    # 1. RMS Error
    if ret < 0.5:
        print(f"✓ Error RMS excelente: {ret:.4f} (< 0.5)")
    elif ret < 1.0:
        print(f"⚠ Error RMS aceptable: {ret:.4f} (0.5-1.0)")
    else:
        print(f"✗ Error RMS alto: {ret:.4f} (> 1.0) - Recalibrar recomendado")
    
    # 2. Reprojection errors
    print(f"\n[ERRORES DE REPROYECCIÓN]")
    print(f"Error medio izquierda: {np.mean(errors_left):.4f} píxeles")
    print(f"Error medio derecha: {np.mean(errors_right):.4f} píxeles")
    print(f"Error máximo izquierda: {np.max(errors_left):.4f} píxeles")
    print(f"Error máximo derecha: {np.max(errors_right):.4f} píxeles")
    
    # Detect outliers (errors > 1.0 pixel)
    outliers_left = [i for i, e in enumerate(errors_left) if e > 1.0]
    outliers_right = [i for i, e in enumerate(errors_right) if e > 1.0]
    
    if outliers_left or outliers_right:
        print(f"\n[WARNING] Outliers detectados:")
        if outliers_left:
            print(f"  Izquierda: imágenes {outliers_left}")
        if outliers_right:
            print(f"  Derecha: imágenes {outliers_right}")
        print(f"  Considera eliminar estas imágenes y recalibrar")
    
    # 3. Fundamental matrix
    if F is not None:
        U, S, Vt = np.linalg.svd(F)
        rank = np.sum(S > 1e-7)
        if rank == 2:
            print(f"\n✓ Matriz fundamental válida (rango = 2)")
        else:
            print(f"\n✗ Matriz fundamental inválida (rango = {rank})")
    
    # 4. Verify resolution
    if img_size[0] != 1280 or img_size[1] != 720:
        print(f"\n⚠ Resolución {img_size} difiere de SEAL (1280x720)")
        print(f"  Las imágenes serán redimensionadas automáticamente")


def print_calibration_summary(
    K_left: np.ndarray,
    K_right: np.ndarray,
    dist_left: np.ndarray,
    dist_right: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    ret: float,
    img_size: Tuple[int, int]
):
    """Print detailed calibration summary"""
    print("\n" + "="*60)
    print("RESUMEN DE CALIBRACIÓN ESTÉREO")
    print("="*60)
    
    print(f"\n📐 RESOLUCIÓN: {img_size[0]}x{img_size[1]}")
    print(f"📊 ERROR RMS: {ret:.4f} píxeles")
    
    print(f"\n🎥 CÁMARA IZQUIERDA (Laser - A):")
    print(f"   Focal: fx={K_left[0,0]:.2f}, fy={K_left[1,1]:.2f}")
    print(f"   Centro: cx={K_left[0,2]:.2f}, cy={K_left[1,2]:.2f}")
    print(f"   Distorsión: k1={dist_left[0,0]:.4f}, k2={dist_left[0,1]:.4f}")
    if dist_left.shape[1] > 4:
        print(f"                k3={dist_left[0,4]:.4f}")
    
    print(f"\n🎥 CÁMARA DERECHA (UV - B):")
    print(f"   Focal: fx={K_right[0,0]:.2f}, fy={K_right[1,1]:.2f}")
    print(f"   Centro: cx={K_right[0,2]:.2f}, cy={K_right[1,2]:.2f}")
    print(f"   Distorsión: k1={dist_right[0,0]:.4f}, k2={dist_right[0,1]:.4f}")
    if dist_right.shape[1] > 4:
        print(f"                k3={dist_right[0,4]:.4f}")
    
    print(f"\n🔄 TRANSFORMACIÓN ESTÉREO:")
    baseline = np.linalg.norm(T)
    print(f"   Baseline: {baseline:.2f} mm")
    print(f"   Traslación: X={T[0,0]:.2f}, Y={T[1,0]:.2f}, Z={T[2,0]:.2f} mm")
    
    # Calculate rotation angles
    rvec, _ = cv2.Rodrigues(R)
    angles = np.degrees(rvec.flatten())
    print(f"   Rotación: X={angles[0]:.2f}°, Y={angles[1]:.2f}°, Z={angles[2]:.2f}°")
    
    print("\n" + "="*60)
