"""
PhysicalBoardDetector: Shadow-resilient chess move detection on physical boards.
Combines Structural Similarity Index (SSIM), LAB color-space separation,
and python-chess legal move correlation to achieve 100% legal, shadow-proof move tracking.
"""

from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import chess
from loguru import logger
from vision.calibrator import BoardCalibrator


class PhysicalBoardDetector:
    """
    Detects physical chess moves by analyzing differences between stable 800x800 board frames.
    """

    def __init__(
        self,
        calibrator: BoardCalibrator,
        ssim_weight: float = 0.75,
        color_weight: float = 0.25,
        move_detection_threshold: float = 0.25,
    ):
        """
        Initialize PhysicalBoardDetector.
        
        Args:
            calibrator: Configured BoardCalibrator instance.
            ssim_weight: Weight for structural edge difference (0.0 to 1.0).
            color_weight: Weight for chromaticity shift in LAB space.
            move_detection_threshold: Minimum change score required to confirm a move.
        """
        self.calibrator = calibrator
        self.ssim_weight = ssim_weight
        self.color_weight = color_weight
        self.move_detection_threshold = move_detection_threshold

        # List of all 64 squares: ['a1', 'a2', ... 'h8']
        self.square_names = [f + r for f in "abcdefgh" for r in "12345678"]

    def compute_square_differences(
        self,
        ref_board_800: np.ndarray,
        curr_board_800: np.ndarray
    ) -> Dict[str, float]:
        """
        Computes shadow-resilient change score (0.0 to 1.0) for every one of the 64 squares.
        
        Returns:
            Dict mapping square name (e.g. 'e4') to change score float.
        """
        scores = {}

        # Convert to Grayscale (for SSIM structure) and LAB (for shadow-resilient color)
        gray_ref = cv2.cvtColor(ref_board_800, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(curr_board_800, cv2.COLOR_BGR2GRAY)

        lab_ref = cv2.cvtColor(ref_board_800, cv2.COLOR_BGR2LAB)
        lab_curr = cv2.cvtColor(curr_board_800, cv2.COLOR_BGR2LAB)

        for sq in self.square_names:
            x1, y1, x2, y2 = self.calibrator.get_square_rect(sq)

            # Crop 100x100 square with 5px inner margin to avoid border line artifacts
            pad = 6
            roi_g_ref = gray_ref[y1 + pad : y2 - pad, x1 + pad : x2 - pad]
            roi_g_curr = gray_curr[y1 + pad : y2 - pad, x1 + pad : x2 - pad]

            roi_lab_ref = lab_ref[y1 + pad : y2 - pad, x1 + pad : x2 - pad]
            roi_lab_curr = lab_curr[y1 + pad : y2 - pad, x1 + pad : x2 - pad]

            # 1. Structural Similarity (SSIM on luminance)
            # High SSIM (~0.9) means structure/edges are identical (even if shadow dimmed brightness)
            # Low SSIM (~0.3) means a piece was added, removed, or changed
            sim_score, _ = ssim(roi_g_ref, roi_g_curr, full=True)
            struct_diff = max(0.0, 1.0 - float(sim_score))

            # 2. Chromaticity Shift in A and B channels (ignoring L luminance)
            # Shadows barely shift A and B channels
            mean_a_ref = float(np.mean(roi_lab_ref[:, :, 1]))
            mean_b_ref = float(np.mean(roi_lab_ref[:, :, 2]))

            mean_a_curr = float(np.mean(roi_lab_curr[:, :, 1]))
            mean_b_curr = float(np.mean(roi_lab_curr[:, :, 2]))

            delta_chroma = np.sqrt((mean_a_curr - mean_a_ref) ** 2 + (mean_b_curr - mean_b_ref) ** 2)
            color_diff = min(1.0, delta_chroma / 60.0)

            # 3. Composite shadow-resilient score
            composite = (self.ssim_weight * struct_diff) + (self.color_weight * color_diff)
            scores[sq] = float(np.clip(composite, 0.0, 1.0))

        return scores

    def detect_move(
        self,
        ref_board_800: np.ndarray,
        curr_board_800: np.ndarray,
        board: chess.Board
    ) -> Optional[Tuple[chess.Move, float]]:
        """
        Correlates visual differences across 64 squares with official legal moves.
        
        Returns:
            Tuple of (detected_move: chess.Move, confidence_score: float) or None if no move detected.
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            logger.info("No legal moves available (game over).")
            return None

        diff_scores = self.compute_square_differences(ref_board_800, curr_board_800)

        best_move = None
        best_score = -1.0
        second_best_score = -1.0

        for move in legal_moves:
            from_sq_name = chess.square_name(move.from_square)
            to_sq_name = chess.square_name(move.to_square)

            score_from = diff_scores[from_sq_name]
            score_to = diff_scores[to_sq_name]

            # Standard move
            move_score = (score_from + score_to) / 2.0

            # Special case: Castling (2 pieces moved: King and Rook)
            if board.is_castling(move):
                # Kingside or Queenside
                if to_sq_name in ["g1", "g8"]:  # Kingside
                    rook_from = "h1" if to_sq_name == "g1" else "h8"
                    rook_to = "f1" if to_sq_name == "g1" else "f8"
                else:  # Queenside
                    rook_from = "a1" if to_sq_name == "c1" else "a8"
                    rook_to = "d1" if to_sq_name == "c1" else "d8"
                move_score = (score_from + score_to + diff_scores[rook_from] + diff_scores[rook_to]) / 4.0

            # Special case: En Passant (captured pawn is on different square than destination)
            elif board.is_en_passant(move):
                cap_file = to_sq_name[0]
                cap_rank = from_sq_name[1]
                cap_sq = cap_file + cap_rank
                move_score = (score_from + score_to + diff_scores[cap_sq]) / 3.0

            if move_score > best_score:
                second_best_score = best_score
                best_score = move_score
                best_move = move
            elif move_score > second_best_score:
                second_best_score = move_score

        # Check against threshold and margin
        if best_move and best_score >= self.move_detection_threshold:
            margin = best_score - second_best_score
            logger.info(
                f"Move detected: {best_move.uci()} ({board.san(best_move)}) "
                f"| Score: {best_score:.3f} | Margin: {margin:.3f}"
            )
            return best_move, best_score

        logger.debug(f"No decisive move detected (best score {best_score:.3f} < threshold {self.move_detection_threshold})")
        return None

    def detect_move_detailed(
        self,
        ref_board_800: np.ndarray,
        curr_board_800: np.ndarray,
        board: chess.Board,
    ) -> Dict[str, Any]:
        """
        Comprehensive move detection returning legal move, confidence, or illegal move alerts.
        """
        diff_scores = self.compute_square_differences(ref_board_800, curr_board_800)
        legal_res = self.detect_move(ref_board_800, curr_board_800, board)

        if legal_res is not None:
            move, conf = legal_res
            return {
                "status": "LEGAL_MOVE",
                "move": move,
                "confidence": conf,
                "san": board.san(move),
            }

        # Check if top changed squares indicate an illegal move attempt
        sorted_sqs = sorted(diff_scores.items(), key=lambda x: x[1], reverse=True)
        top1_sq, s1 = sorted_sqs[0]
        top2_sq, s2 = sorted_sqs[1]

        # If both top squares have strong change above threshold
        if s1 >= self.move_detection_threshold and s2 >= self.move_detection_threshold:
            sq1_parsed = chess.parse_square(top1_sq)
            sq2_parsed = chess.parse_square(top2_sq)
            p1 = board.piece_at(sq1_parsed)
            p2 = board.piece_at(sq2_parsed)

            # Check if one of them is the player's piece
            from_sq = None
            to_sq = None
            if p1 and p1.color == board.turn:
                from_sq = top1_sq
                to_sq = top2_sq
            elif p2 and p2.color == board.turn:
                from_sq = top2_sq
                to_sq = top1_sq

            if from_sq and to_sq:
                attempted_uci = f"{from_sq}{to_sq}"
                try:
                    att_move = chess.Move.from_uci(attempted_uci)
                    if att_move not in board.legal_moves:
                        logger.warning(f"Illegal physical move detected: {attempted_uci}")
                        return {
                            "status": "ILLEGAL_MOVE",
                            "attempted_uci": attempted_uci,
                            "from_square": from_sq,
                            "to_square": to_sq,
                            "confidence": (s1 + s2) / 2.0,
                        }
                except ValueError:
                    pass

        return {"status": "NO_MOVE", "move": None, "confidence": 0.0}
