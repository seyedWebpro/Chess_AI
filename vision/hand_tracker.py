"""
HandTracker: Real-time hand occlusion detection using MediaPipe Hands.
Detects when human hands are over the chessboard and locks move detection
until the hand has completely exited the playing area.
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np
from loguru import logger


class HandTracker:
    """
    Detects human hands over the chessboard to prevent false move triggers during physical moves.
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
    ):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._mp_hands = None
        self._hands = None
        self._mp_draw = None

        self._init_mediapipe()

    def _init_mediapipe(self) -> None:
        """Initializes MediaPipe Hands solution if available."""
        try:
            import mediapipe as mp
            self._mp_hands = mp.solutions.hands
            self._mp_draw = mp.solutions.drawing_utils
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            logger.info("MediaPipe Hands initialized successfully.")
        except Exception as e:
            logger.warning(f"MediaPipe Hands not available ({e}). Using skin/motion fallback.")
            self._hands = None

    def find_hands(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Detects hand landmark coordinates in frame.
        
        Returns:
            List of hand landmarks arrays with shape (21, 2) in pixel coordinates [x, y].
        """
        if self._hands is None:
            return self._fallback_hand_contours(frame)

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb_frame)

        hand_landmarks_list = []
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                pts = []
                for lm in hand_lms.landmark:
                    px = int(round(lm.x * w))
                    py = int(round(lm.y * h))
                    pts.append([px, py])
                hand_landmarks_list.append(np.array(pts, dtype=np.int32))

        return hand_landmarks_list

    def is_hand_over_board(
        self,
        frame: np.ndarray,
        board_corners: np.ndarray,
        padding_px: int = 20
    ) -> bool:
        """
        Determines whether any hand is currently hovering over or touching the board area.
        
        Args:
            frame: Camera frame (e.g. 1280x720).
            board_corners: 4 corners of the board in camera frame [TL, TR, BR, BL].
            padding_px: Outer margin around the board.
            
        Returns:
            True if hand is inside/overlapping board region, False if board is clear.
        """
        if board_corners is None or len(board_corners) != 4:
            return False

        hand_pts_list = self.find_hands(frame)
        if not hand_pts_list:
            return False

        # Build board polygon contour
        board_cnt = board_corners.astype(np.int32).reshape((-1, 1, 2))

        for hand_pts in hand_pts_list:
            # Check wrist (pt 0) and fingertips (pts 4, 8, 12, 16, 20)
            critical_pts = hand_pts[[0, 4, 8, 12, 16, 20]]
            for pt in critical_pts:
                dist = cv2.pointPolygonTest(board_cnt, (float(pt[0]), float(pt[1])), measureDist=True)
                # If distance > -padding_px, point is inside or near the board
                if dist >= -padding_px:
                    return True

        return False

    def _fallback_hand_contours(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Fallback skin-color detector when MediaPipe is not present.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hand_pts = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 2500:  # Hand-sized blob
                hull = cv2.convexHull(cnt)
                hand_pts.append(hull.reshape(-1, 2))

        return hand_pts

    def draw_hands(self, frame: np.ndarray, hand_pts_list: List[np.ndarray]) -> np.ndarray:
        """Visualizes detected hands with skeletons / outlines."""
        annotated = frame.copy()
        for pts in hand_pts_list:
            if len(pts) == 21:  # MediaPipe 21 landmarks
                for p in pts:
                    cv2.circle(annotated, (p[0], p[1]), 5, (0, 255, 255), -1)
            else:
                cv2.drawContours(annotated, [pts], -1, (0, 255, 255), 2)
        return annotated

    def close(self) -> None:
        """Releases MediaPipe resources."""
        if self._hands:
            self._hands.close()
            self._hands = None
