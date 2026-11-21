"""
ScreenBoardDetector: Automatic board localization, theme identification,
and orientation detection on digital screens (Chess.com / Lichess).
"""

from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import cv2
import chess
from loguru import logger


class ScreenBoardDetector:
    """
    Detects and localizes an online chessboard within a desktop screen capture.
    """

    # Popular Online Themes (BGR format)
    THEMES = {
        "chess_com_green": {
            "light": (208, 236, 235),  # #ebecd0
            "dark": (86, 149, 119),     # #779556
            "highlight_light": (130, 246, 245),  # #f5f682 (Yellow)
            "highlight_dark": (67, 202, 185),    # #b9ca43 (Olive yellow)
        },
        "lichess_blue": {
            "light": (230, 227, 222),  # #dee3e6
            "dark": (173, 162, 140),    # #8ca2ad
            "highlight_light": (100, 210, 210),
            "highlight_dark": (80, 180, 160),
        },
        "lichess_brown": {
            "light": (181, 217, 240),  # #f0d9b5
            "dark": (99, 136, 181),     # #b58863
            "highlight_light": (100, 210, 210),
            "highlight_dark": (80, 180, 160),
        },
    }

    def __init__(
        self,
        board_bbox: Optional[Tuple[int, int, int, int]] = None,
        white_bottom: bool = True,
        theme_name: str = "chess_com_green",
    ):
        """
        Initialize ScreenBoardDetector.
        
        Args:
            board_bbox: Optional tuple (x, y, width, height) of chessboard on screen.
            white_bottom: True if White is at bottom (standard view), False if Black is at bottom.
            theme_name: Name of color theme preset.
        """
        self.board_bbox = board_bbox  # (x, y, w, h)
        self.white_bottom = white_bottom
        self.theme_name = theme_name
        self.active_theme = self.THEMES.get(theme_name, self.THEMES["chess_com_green"])

        self.files = ["a", "b", "c", "d", "e", "f", "g", "h"]
        self.ranks = ["1", "2", "3", "4", "5", "6", "7", "8"]

    def set_board_bbox(self, x: int, y: int, width: int, height: int):
        """Manually sets the chessboard bounding box."""
        self.board_bbox = (int(x), int(y), int(width), int(height))
        logger.info(f"Board bbox set manually: {self.board_bbox}")

    def find_board_auto(
        self,
        screen_frame: np.ndarray,
        min_size: int = 350,
        max_size: int = 1200,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Automatically locates the chessboard on screen by finding square contours
        with 8x8 checkerboard symmetry.
        
        Returns:
            Tuple (x, y, width, height) or None if not found.
        """
        h_screen, w_screen = screen_frame.shape[:2]
        gray = cv2.cvtColor(screen_frame, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        best_bbox = None
        best_aspect_score = 999.0

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < min_size or h < min_size or w > max_size or h > max_size:
                continue

            # Aspect ratio check (must be near 1.0)
            aspect = float(w) / float(h)
            diff_from_square = abs(aspect - 1.0)

            if diff_from_square < 0.05:  # within 5% of square
                if diff_from_square < best_aspect_score:
                    # Verify 8x8 checkerboard property inside
                    roi = gray[y:y+h, x:x+w]
                    if self._verify_checkerboard_pattern(roi):
                        best_aspect_score = diff_from_square
                        best_bbox = (x, y, w, h)

        if best_bbox:
            self.board_bbox = best_bbox
            logger.success(f"Chessboard auto-localized at: x={best_bbox[0]}, y={best_bbox[1]}, size={best_bbox[2]}x{best_bbox[3]}")
            return best_bbox

        logger.warning("Auto board localization did not find a conclusive 8x8 board.")
        return None

    def _verify_checkerboard_pattern(self, roi_gray: np.ndarray) -> bool:
        """Checks if a square ROI exhibits 8x8 alternating light/dark square characteristics."""
        h, w = roi_gray.shape
        sq_h, sq_w = h // 8, w // 8
        if sq_h < 30 or sq_w < 30:
            return False

        # Sample centers of first 4 squares on rank 1
        means = []
        for c in range(4):
            sq_roi = roi_gray[0:sq_h, c*sq_w:(c+1)*sq_w]
            means.append(np.mean(sq_roi))

        # Check alternating pattern: diff(sq0, sq1) should be significant
        if abs(means[0] - means[1]) > 20 and abs(means[1] - means[2]) > 20:
            return True
        return False

    def detect_orientation(self, board_crop: np.ndarray) -> bool:
        """
        Detects if White is at bottom (standard) or Black is at bottom.
        Analyzes the bottom-most rank (rank 1 if white bottom, rank 8 if black bottom).
        Returns True if White is at bottom.
        """
        h, w = board_crop.shape[:2]
        sq_h = h // 8

        # Sample bottom rank and top rank
        bottom_rank_roi = board_crop[7 * sq_h:8 * sq_h, :]
        top_rank_roi = board_crop[0:sq_h, :]

        # White pieces are significantly brighter in luminance than black pieces
        bottom_lum = np.mean(cv2.cvtColor(bottom_rank_roi, cv2.COLOR_BGR2GRAY))
        top_lum = np.mean(cv2.cvtColor(top_rank_roi, cv2.COLOR_BGR2GRAY))

        # Typically bottom rank has White pieces if playing as White
        self.white_bottom = bool(bottom_lum >= top_lum)
        logger.info(f"Board orientation detected: {'White Bottom' if self.white_bottom else 'Black Bottom'}")
        return self.white_bottom

    def crop_board(self, screen_frame: np.ndarray) -> Optional[np.ndarray]:
        """Crops the chessboard from full screen capture."""
        if self.board_bbox is None:
            return None
        x, y, w, h = self.board_bbox
        return screen_frame[y:y+h, x:x+w].copy()

    def get_square_rect(self, square_name: str, board_size: int) -> Tuple[int, int, int, int]:
        """
        Returns (x1, y1, x2, y2) bounds of square_name within the cropped board_size x board_size.
        """
        sq = square_name.lower().strip()
        file_char = sq[0]
        rank_char = sq[1]

        sq_size = board_size // 8

        col = self.files.index(file_char) if self.white_bottom else 7 - self.files.index(file_char)
        rank_idx = int(rank_char) - 1
        row = (7 - rank_idx) if self.white_bottom else rank_idx

        x1 = col * sq_size
        y1 = row * sq_size
        x2 = x1 + sq_size
        y2 = y1 + sq_size

        return x1, y1, x2, y2

    def get_square_center(self, square_name: str, board_size: int) -> Tuple[int, int]:
        """Returns (cx, cy) pixel center of square within cropped board image."""
        x1, y1, x2, y2 = self.get_square_rect(square_name, board_size)
        return (x1 + x2) // 2, (y1 + y2) // 2

    def get_screen_square_center(self, square_name: str) -> Optional[Tuple[int, int]]:
        """
        Returns absolute (x_screen, y_screen) monitor coordinates of the square center.
        Used for mouse automation (WindMouse in Phase 9).
        """
        if self.board_bbox is None:
            return None
        bx, by, bw, bh = self.board_bbox
        cx, cy = self.get_square_center(square_name, bw)
        return bx + cx, by + cy
