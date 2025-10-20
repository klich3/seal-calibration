"""ChArUco board pattern detector (following reference implementation)"""
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
        dictionary_id: int = cv2.aruco.DICT_6X6_250
    ):
        """
        Initialize ChArUco detector (following reference code pattern)
        
        Args:
            squares_x: Number of squares in X direction (columns)
            squares_y: Number of squares in Y direction (rows)
            square_length: Square side length in mm (or meters in reference: 0.04 = 40mm)
            marker_length: Marker side length in mm (or meters in reference: 0.02 = 20mm)
            dictionary_id: ArUco dictionary ID (reference uses DICT_6X6_250)
        """
        self.squares_x = squares_x
        self.squares_y = squares_y
        self.square_length = square_length
        self.marker_length = marker_length
        
        # Create ArUco dictionary and ChArUco board (exactly like reference)
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.charuco_board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            square_length,
            marker_length,
            self.dictionary
        )
        
        # Create detector parameters (exactly like reference)
        self.params = cv2.aruco.DetectorParameters()
        self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        
        # Sub-pixel corner detection criterion (exactly like reference)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
    
    def detect(self, gray_frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect ChArUco board in grayscale image (following reference implementation)
        
        Args:
            gray_frame: Grayscale image
            
        Returns:
            Tuple of (found, charuco_corners, charuco_ids)
        """
        try:
            # Detect markers and corners (exactly like reference code)
            corners, ids, rejected = cv2.aruco.detectMarkers(gray_frame, self.dictionary, parameters=self.params)
            
            allCorners = None
            allIds = None
            ret = False
            
            if corners is not None and len(corners) > 0:
                # Apply sub-pixel refinement to marker corners
                for corner in corners:
                    cv2.cornerSubPix(gray_frame, corner, (3, 3), (-1, -1), self.criteria)
                
                # Interpolate ChArUco corners
                ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray_frame, self.board
                )
                
                if charuco_ids is not None and len(charuco_ids) > 0:
                    allCorners = charuco_corners
                    allIds = charuco_ids
                    ret = True
            
            return ret, allCorners, allIds
            
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
            objp.append(self.charuco_board.getChessboardCorners()[corner_id])
        
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
        board_image = self.charuco_board.generateImage(size_pixels)
        cv2.imwrite(output_path, board_image)
        print(f"[INFO] ChArUco board saved to {output_path}")
