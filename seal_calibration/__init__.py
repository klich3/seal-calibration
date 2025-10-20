"""
SEAL Calibration Library
========================

Calibration library for 3DMakerPro SEAL 3D scanners.
Provides tools for camera calibration, stereo calibration, and projector calibration.
"""

from .__version__ import __version__

from .models.camera_params import CameraParams
from .models.stereo_params import StereoParams
from .models.seal_calib import SEALCalibration

from .core.camera import CameraCalibrator
from .core.stereo import StereoCalibrator
from .core.projector import ProjectorCalibrator

from .io.loader import SEALCalibrationLoader
from .io.writer import SEALCalibrationWriter

__all__ = [
    "__version__",
    "CameraParams",
    "StereoParams",
    "SEALCalibration",
    "CameraCalibrator",
    "StereoCalibrator",
    "ProjectorCalibrator",
    "SEALCalibrationLoader",
    "SEALCalibrationWriter",
]
