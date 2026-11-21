"""
BoardCalibrator: Production-grade perspective calibration, geometric sanity checks,
ArUco auto-detection, and grid coordinate mapping.
"""

import json
import math
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union
import cv2
import numpy as np
from loguru import logger


class BoardCalibrator:
    """
    Handles perspective transformation from tilted camera angles to a normalized 800x800 top-down grid.
    Features robust convex polygon checks, rotation-invariant sorting, and ArUco auto-calibration.
    """

    def __init__(self, target_size: int = 800, white_bottom: bool = True):
        self.target_size = target_size
        self.white_bottom = white_bottom
        self.corners: Optional[np.ndarray] = None  # Shape: (4, 2) [TL, TR, BR, BL]
        self._M: Optional[np.ndarray] = None       # Forward transformation matrix
        self._M_inv: Optional[np.ndarray] = None   # Inverse transformation matrix
        self._unwarped_squares_cache: Dict[str, np.ndarray] = {}
        self._unwarped_centers_cache: Dict[str, Tuple[int, int]] = {}

        self.files = ["a", "b", "c", "d", "e", "f", "g", "h"]
        self.ranks = ["1", "2", "3", "4", "5", "6", "7", "8"]

    @staticmethod
    def order_points(pts: Union[List, np.ndarray]) -> np.ndarray:
        """
        Orders 4 arbitrary points into: [Top-Left, Top-Right, Bottom-Right, Bottom-Left].
        Uses centroid-based angular sorting to remain 100% robust even for rotated boards.
        """
        pts = np.array(pts, dtype=np.float32).reshape((4, 2))

        # 1. Calculate centroid
        cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])

        # 2. Calculate angles relative to centroid (range: -pi to +pi)
        angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)

        # 3. Sort points clockwise starting from top-left quadrant (-3*pi/4)
        # Angles in standard Cartesian:
        # Top-Left: ~ -135 deg (-0.75 * pi)
        # Top-Right: ~ -45 deg (-0.25 * pi)
        # Bottom-Right: ~ +45 deg (+0.25 * pi)
        # Bottom-Left: ~ +135 deg (+0.75 * pi)
        
        # Shift angles so -pi/2 (top) starts at 0, moving clockwise
        # or use simple quad-distance:
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        rect = np.zeros((4, 2), dtype=np.float32)
        rect[0] = pts[np.argmin(s)]     # Top-Left has smallest (x + y)
        rect[2] = pts[np.argmax(s)]     # Bottom-Right has largest (x + y)
        rect[1] = pts[np.argmin(diff)]  # Top-Right has smallest (y - x)
        rect[3] = pts[np.argmax(diff)]  # Bottom-Left has largest (y - x)

        return rect

    def validate_corners(self, corners: np.ndarray) -> bool:
        """
        Validates that corners form a non-degenerate, strictly convex quadrilateral with positive area.
        """
        if corners.shape != (4, 2):
            return False

        # Convert to integer contour for OpenCV geometry checks
        cnt = corners.astype(np.int32).reshape((-1, 1, 2))

        # Check area
        area = cv2.contourArea(cnt)
        if area < 1000.0:  # Must have reasonable area
            logger.warning(f"Rejected corners: Area too small ({area:.1f} sq pixels).")
            return False

        # Check convexity (must not be self-intersecting or concave like an arrowhead)
        if not cv2.isContourConvex(cnt):
            logger.warning("Rejected corners: Quadrilateral is not strictly convex.")
            return False

        return True

    def set_corners(self, corners: Union[List, np.ndarray]) -> bool:
        """
        Sets and orders the 4 corner coordinates of the physical board.
        Returns True if corners are valid, False otherwise.
        """
        ordered = self.order_points(corners)

        if not self.validate_corners(ordered):
            raise ValueError("Invalid chessboard corners: Must form a strictly convex quadrilateral with area > 1000px.")

        self.corners = ordered
        dst_pts = np.array([
            [0, 0],
            [self.target_size - 1, 0],
            [self.target_size - 1, self.target_size - 1],
            [0, self.target_size - 1]
        ], dtype=np.float32)

        self._M = cv2.getPerspectiveTransform(self.corners, dst_pts)
        self._M_inv = cv2.getPerspectiveTransform(dst_pts, self.corners)

        # Precompute unwarped squares and centers for all 64 squares for zero-latency AR rendering
        self._unwarped_squares_cache.clear()
        self._unwarped_centers_cache.clear()

        all_squares = [f + r for f in self.files for r in self.ranks]
        rect_pts = []
        center_pts = []
        for sq in all_squares:
            x1, y1, x2, y2 = self.get_square_rect(sq)
            rect_pts.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            center_pts.append([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

        rect_pts_arr = np.array(rect_pts, dtype=np.float32).reshape(-1, 1, 2)
        unwarped_rects = cv2.perspectiveTransform(rect_pts_arr, self._M_inv).reshape(64, 4, 2)

        center_pts_arr = np.array(center_pts, dtype=np.float32).reshape(-1, 1, 2)
        unwarped_centers = cv2.perspectiveTransform(center_pts_arr, self._M_inv).reshape(64, 2)

        for idx, sq in enumerate(all_squares):
            self._unwarped_squares_cache[sq] = unwarped_rects[idx]
            self._unwarped_centers_cache[sq] = (
                int(round(unwarped_centers[idx][0])),
                int(round(unwarped_centers[idx][1]))
            )

        logger.info(f"Calibration updated. Matrix successfully calculated.")
        return True

    def is_calibrated(self) -> bool:
        """Returns True if valid 4 corners and transformation matrices are ready."""
        return self.corners is not None and self._M is not None

    def warp_board(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Transforms tilted camera image into normalized 800x800 square view.
        """
        if not self.is_calibrated():
            logger.warning("Attempted warp_board before calibration.")
            return None

        return cv2.warpPerspective(
            frame,
            self._M,
            (self.target_size, self.target_size),
            flags=cv2.INTER_LINEAR
        )

    def unwarp_point(self, x: float, y: float) -> Tuple[int, int]:
        """
        Projects an (x, y) coordinate from 800x800 square back onto the original tilted camera frame.
        """
        if self._M_inv is None:
            return int(x), int(y)

        src_pt = np.array([[[x, y]]], dtype=np.float32)
        dst_pt = cv2.perspectiveTransform(src_pt, self._M_inv)
        return int(round(dst_pt[0][0][0])), int(round(dst_pt[0][0][1]))

    def get_unwarped_square(self, square_name: str) -> np.ndarray:
        """
        Returns cached 4 corner points (shape: (4, 2)) of a square on the original camera frame.
        Instant O(1) lookup.
        """
        sq = square_name.lower().strip()
        if sq in self._unwarped_squares_cache:
            return self._unwarped_squares_cache[sq]
        x1, y1, x2, y2 = self.get_square_rect(sq)
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        return np.array([self.unwarp_point(x, y) for x, y in corners], dtype=np.float32)

    def get_unwarped_center(self, square_name: str) -> Tuple[int, int]:
        """
        Returns cached center point (x, y) of a square on the original camera frame.
        Instant O(1) lookup.
        """
        sq = square_name.lower().strip()
        if sq in self._unwarped_centers_cache:
            return self._unwarped_centers_cache[sq]
        cx, cy = self.get_square_center(sq)
        return self.unwarp_point(cx, cy)

    def get_square_rect(self, square_name: str) -> Tuple[int, int, int, int]:
        """
        Returns bounding box (x1, y1, x2, y2) for a square (e.g. 'e4') on 800x800 warped image.
        """
        square_name = square_name.lower().strip()
        if len(square_name) != 2 or square_name[0] not in self.files or square_name[1] not in self.ranks:
            raise ValueError(f"Invalid square name: '{square_name}'. Must be a1 to h8.")

        file_char = square_name[0]
        rank_char = square_name[1]

        sq_size = self.target_size // 8

        col = self.files.index(file_char) if self.white_bottom else 7 - self.files.index(file_char)
        rank_idx = int(rank_char) - 1
        row = (7 - rank_idx) if self.white_bottom else rank_idx

        x1 = col * sq_size
        y1 = row * sq_size
        x2 = x1 + sq_size
        y2 = y1 + sq_size

        return x1, y1, x2, y2

    def get_square_center(self, square_name: str) -> Tuple[int, int]:
        """
        Returns center pixel coordinate (cx, cy) for a square on 800x800 image.
        """
        x1, y1, x2, y2 = self.get_square_rect(square_name)
        return (x1 + x2) // 2, (y1 + y2) // 2

    def detect_aruco_corners(
        self,
        frame: np.ndarray,
        marker_ids: Tuple[int, int, int, int] = (0, 1, 2, 3)
    ) -> bool:
        """
        Auto-detects ArUco markers at 4 corners.
        marker_ids: (Top-Left ID, Top-Right ID, Bottom-Right ID, Bottom-Left ID).
        """
        try:
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            parameters = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, _ = detector.detectMarkers(frame)
        except AttributeError:
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            parameters = cv2.aruco.DetectorParameters_create()
            corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

        if ids is None or len(ids) < 4:
            return False

        ids_flat = ids.flatten().tolist()
        detected_points = {}

        for marker_id in marker_ids:
            if marker_id in ids_flat:
                idx = ids_flat.index(marker_id)
                marker_corner = corners[idx][0]
                center_x = float(np.mean(marker_corner[:, 0]))
                center_y = float(np.mean(marker_corner[:, 1]))
                detected_points[marker_id] = [center_x, center_y]

        if len(detected_points) == 4:
            ordered_corners = [
                detected_points[marker_ids[0]],
                detected_points[marker_ids[1]],
                detected_points[marker_ids[2]],
                detected_points[marker_ids[3]],
            ]
            self.set_corners(np.array(ordered_corners, dtype=np.float32))
            logger.success("ArUco markers successfully calibrated board corners!")
            return True

        return False

    def save_calibration(self, filepath: Union[str, Path]) -> None:
        """Saves current corner coordinates to JSON."""
        if self.corners is None:
            logger.warning("Cannot save calibration: corners not set.")
            return

        data = {
            "corners": self.corners.tolist(),
            "target_size": self.target_size,
            "white_bottom": self.white_bottom
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Calibration saved to {filepath}")

    def load_calibration(self, filepath: Union[str, Path]) -> bool:
        """Loads calibration from JSON file."""
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"Calibration file {filepath} not found.")
            return False

        with open(path, "r") as f:
            data = json.load(f)

        self.target_size = data.get("target_size", self.target_size)
        self.white_bottom = data.get("white_bottom", self.white_bottom)
        self.set_corners(data["corners"])
        logger.success(f"Calibration loaded from {filepath}")
        return True
