"""ChArUco board pattern detector"""
import cv2
import numpy as np
from typing import Tuple, Optional


class CharucoDetector:
    """Detects ChArUco board patterns"""
    
    def __init__(
        self, 
        squares_x: int, 
        squares_y: int, 
        square_length: float, 
        marker_length: float,
        dictionary_id: int = cv2.aruco.DICT_4X4_250
    ):
        """
        Initialize ChArUco detector
        
        Args:
            squares_x: Number of squares in X direction
            squares_y: Number of squares in Y direction
            square_length: Square side length in mm
            marker_length: Marker side length in mm
            dictionary_id: ArUco dictionary ID
        """
        self.squares_x = squares_x
        self.squares_y = squares_y
        self.square_length = square_length
        self.marker_length = marker_length
        
        # Create ArUco dictionary and ChArUco board
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            square_length,
            marker_length,
            self.dictionary
        )
        
        # Create detector with default parameters
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)
    
    def detect(self, gray_frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect ChArUco board in grayscale image
        
        Args:
            gray_frame: Grayscale image
            
        Returns:
            Tuple of (found, charuco_corners, charuco_ids)
        """
        try:
            # Detect ArUco markers
            corners, ids, rejected = self.detector.detectMarkers(gray_frame)
            
            if ids is not None and len(ids) > 0:
                # Interpolate ChArUco corners
                num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    corners,
                    ids,
                    gray_frame,
                    self.board
                )
                
                if num_corners > 0:
                    # ChArUco corners are already interpolated, no need for cornerSubPix
                    return True, charuco_corners, charuco_ids
            
            return False, None, None
            
        except Exception as e:
            print(f"[WARNING] ChArUco detection error: {str(e)}")
            return False, None, None
    
    def get_object_points(
        self, 
        charuco_ids: np.ndarray
    ) -> np.ndarray:
        """
        Get 3D object points for detected ChArUco corners
        
        Args:
            charuco_ids: IDs of detected ChArUco corners
            
        Returns:
            Array of 3D object points
        """
        objp = []
        for corner_id in charuco_ids.flatten():
            objp.append(self.board.chessboardCorners[corner_id])
        
        return np.array(objp, dtype=np.float32)
    
    def draw_corners(
        self, 
        image: np.ndarray, 
        charuco_corners: np.ndarray, 
        charuco_ids: np.ndarray
    ) -> np.ndarray:
        """
        Draw detected ChArUco corners on image
        
        Args:
            image: Image to draw on
            charuco_corners: Detected corners
            charuco_ids: Corner IDs
            
        Returns:
            Image with drawn corners
        """
        if charuco_corners is not None and charuco_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(
                image,
                charuco_corners,
                charuco_ids,
                (0, 255, 0)
            )
        
        return image
    
    def generate_board(self, output_path: str, size_pixels: Tuple[int, int] = (2000, 2000)):
        """
        Generate and save ChArUco board image
        
        Args:
            output_path: Path to save the board image
            size_pixels: Board size in pixels (width, height)
        """
        board_image = self.board.generateImage(size_pixels)
        cv2.imwrite(output_path, board_image)
        print(f"[INFO] ChArUco board saved to {output_path}")
