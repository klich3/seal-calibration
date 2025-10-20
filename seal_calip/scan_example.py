#!/usr/bin/env python3
"""Ejemplo de uso del escáner SEAL con calibración"""
from seal_scanner import SEALScanner
from seal_calibration.io import SEALCalibrationLoader


def main():
    # Ruta al archivo de calibración
    calibration_file = "stereo_calibration_seal.txt"
    
    # Crear escáner con archivo de calibración
    print("Inicializando escáner...")
    scanner = SEALScanner(calibration_file=calibration_file)
    
    # Iniciar escaneo
    print("Iniciando escaneo...")
    scanner.start_scan()
    
    # En una aplicación real, aquí capturaríamos frames
    # while scanning:
    #     img_laser, img_uv = scanner.camera.capture()
    #     pcd = scanner.process_frame(img_laser, img_uv)
    #     # Visualizar pcd en tiempo real
    
    # Finalizar y exportar
    output_file = "scan_output.ply"
    print(f"Finalizando escaneo y exportando a {output_file}...")
    scanner.finalize_scan(output_file)
    
    print("¡Escaneo completado!")


if __name__ == "__main__":
    main()
