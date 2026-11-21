"""
ScreenMoveDetector: Sub-millisecond move detection on Chess.com and Lichess.
Features:
- Move Highlight Color Detector (Yellow/Olive hue analysis) for sub-1ms move extraction.
- Universal legal move correlation with python-chess.
- Theme-aware square sampling (samples corners to avoid piece obstruction).
- Pixel differencing fallback for custom themes.
- Re-trigger protection: Only confirms new moves.
"""

from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import cv2
import chess
from loguru import logger
from screen.screen_board_detector import ScreenBoardDetector


class ScreenMoveDetector:
    """
    Detects online chess moves from screen captures using highlight color analysis.
    """

    def __init__(
        self,
        board_detector: ScreenBoardDetector,
        highlight_threshold: float = 0.35,
    ):
        """
        Initialize ScreenMoveDetector.
        
        Args:
            board_detector: Configured ScreenBoardDetector instance.
            highlight_threshold: Minimum highlight score required to confirm a move.
        """
        self.detector = board_detector
        self.highlight_threshold = highlight_threshold
        self.last_confirmed_move: Optional[chess.Move] = None
        self._cached_board_size: int = 0
        self._cached_white_bottom: Optional[bool] = None
        self._cached_y_coords: Optional[np.ndarray] = None
        self._cached_x_coords: Optional[np.ndarray] = None
        self._cached_base_colors: Optional[np.ndarray] = None
        self._cached_sq_names: List[str] = []

    def compute_square_highlight_scores(self, board_crop: np.ndarray) -> Dict[str, float]:
        """
        Computes the highlight intensity score for all 64 squares using high-speed
        vectorized Euclidean distance delta from baseline square colors.
        Runs in ~0.2ms.
        """
        h, w = board_crop.shape[:2]

        # Recompute coordinate cache if board dimensions or orientation changed
        if (
            h != self._cached_board_size
            or self.detector.white_bottom != self._cached_white_bottom
            or self._cached_y_coords is None
        ):
            sq_size = h // 8
            offset = max(6, sq_size // 6)
            y_coords: List[int] = []
            x_coords: List[int] = []
            base_colors: List[np.ndarray] = []
            sq_names: List[str] = []

            theme_light = np.array(self.detector.active_theme["light"], dtype=np.float32)
            theme_dark = np.array(self.detector.active_theme["dark"], dtype=np.float32)

            for r in range(8):
                for c in range(8):
                    file_idx = c if self.detector.white_bottom else 7 - c
                    rank_idx = (7 - r) if self.detector.white_bottom else r
                    sq_name = self.detector.files[file_idx] + self.detector.ranks[rank_idx]
                    sq_names.append(sq_name)

                    # Chess square coloring: (file + rank) is odd -> light square, even -> dark square
                    is_light = ((file_idx + rank_idx) % 2 != 0)
                    base = theme_light if is_light else theme_dark
                    base_colors.append(base)

                    # 4 corners per square (to avoid piece obstruction in the center)
                    y_coords.extend([
                        r * sq_size + offset,
                        r * sq_size + offset,
                        (r + 1) * sq_size - offset,
                        (r + 1) * sq_size - offset,
                    ])
                    x_coords.extend([
                        c * sq_size + offset,
                        (c + 1) * sq_size - offset,
                        c * sq_size + offset,
                        (c + 1) * sq_size - offset,
                    ])

            self._cached_board_size = h
            self._cached_white_bottom = self.detector.white_bottom
            self._cached_y_coords = np.array(y_coords, dtype=np.int32)
            self._cached_x_coords = np.array(x_coords, dtype=np.int32)
            self._cached_base_colors = np.array(base_colors, dtype=np.float32)
            self._cached_sq_names = sq_names

        # Sample 256 corner points across all 64 squares in a single fast array lookup
        samples = board_crop[self._cached_y_coords, self._cached_x_coords].reshape(64, 4, 3).astype(np.float32)
        means = np.mean(samples, axis=1)  # shape (64, 3)
        diff = means - self._cached_base_colors
        dist = np.sqrt(np.sum(diff ** 2, axis=-1)) / 100.0  # normalized delta score

        scores = {self._cached_sq_names[i]: float(dist[i]) for i in range(64)}
        return scores

    def detect_move_from_highlights(
        self,
        board_crop: np.ndarray,
        board: chess.Board,
    ) -> Optional[Tuple[chess.Move, float]]:
        """
        Extracts the played move by identifying the two highlighted squares and
        correlating them with board.legal_moves.
        
        Returns:
            Tuple of (chess.Move, confidence) or None if no new move is present.
        """
        scores = self.compute_square_highlight_scores(board_crop)

        # Sort squares by highlight score
        sorted_sqs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top1_sq, s1 = sorted_sqs[0]
        top2_sq, s2 = sorted_sqs[1]

        # Minimum threshold check
        if s1 < self.highlight_threshold or s2 < self.highlight_threshold:
            return None

        # Try matching legal moves between top highlighted squares
        candidate_moves = []
        for move in board.legal_moves:
            from_name = chess.square_name(move.from_square)
            to_name = chess.square_name(move.to_square)

            if from_name in [top1_sq, top2_sq] and to_name in [top1_sq, top2_sq]:
                combined_score = (scores[from_name] + scores[to_name]) / 2.0
                if move.promotion == chess.QUEEN:
                    combined_score += 0.001
                candidate_moves.append((move, combined_score))

            # Castling case (King + Rook squares)
            elif board.is_castling(move):
                if from_name in [top1_sq, top2_sq] or to_name in [top1_sq, top2_sq]:
                    candidate_moves.append((move, s1))

        if not candidate_moves:
            logger.debug(f"Highlighted squares ({top1_sq}: {s1:.2f}, {top2_sq}: {s2:.2f}) do not match legal moves.")
            return None

        # Pick best candidate move
        candidate_moves.sort(key=lambda x: x[1], reverse=True)
        best_move, best_conf = candidate_moves[0]

        # Check if identical to last confirmed move to prevent double triggers
        if self.last_confirmed_move and best_move == self.last_confirmed_move:
            return None

        self.last_confirmed_move = best_move
        logger.info(f"Online Move Detected: {best_move.uci()} ({board.san(best_move)}) | Conf: {best_conf:.2f}")
        return best_move, best_conf

    def detect_move_from_difference(
        self,
        prev_crop: np.ndarray,
        curr_crop: np.ndarray,
        board: chess.Board,
    ) -> Optional[Tuple[chess.Move, float]]:
        """
        Fallback difference detector if screen theme does not use yellow/green highlights.
        """
        h, w = curr_crop.shape[:2]
        diff = cv2.absdiff(curr_crop, prev_crop)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        scores: Dict[str, float] = {}
        for sq in chess.SQUARES:
            sq_name = chess.square_name(sq)
            x1, y1, x2, y2 = self.detector.get_square_rect(sq_name, h)
            scores[sq_name] = float(np.mean(gray_diff[y1:y2, x1:x2])) / 255.0

        best_move = None
        best_score = -1.0
        for move in board.legal_moves:
            f = chess.square_name(move.from_square)
            t = chess.square_name(move.to_square)
            score = (scores[f] + scores[t]) / 2.0
            if score > best_score:
                best_score = score
                best_move = move

        if best_move and best_score > 0.08:
            if self.last_confirmed_move and best_move == self.last_confirmed_move:
                return None
            self.last_confirmed_move = best_move
            return best_move, best_score

        return None
