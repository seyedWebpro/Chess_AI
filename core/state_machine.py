"""
GameStateManager: Finite State Machine (FSM) managing the physical game loop.
Coordinates HandTracker, BoardCalibrator, PhysicalBoardDetector, and EngineManager.
"""

from enum import Enum
import time
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np
import chess
from loguru import logger

from core.engine_manager import EngineManager
from vision.calibrator import BoardCalibrator
from vision.hand_tracker import HandTracker
from vision.board_detector import PhysicalBoardDetector
from audio.voice_assistant import VoiceAssistant


class GameStatus(Enum):
    IDLE = "IDLE"
    WAITING_FOR_MOVE = "WAITING_FOR_MOVE"
    HAND_DETECTED = "HAND_DETECTED"
    STABILIZING = "STABILIZING"
    MOVE_CONFIRMED = "MOVE_CONFIRMED"
    ILLEGAL_MOVE_ATTEMPTED = "ILLEGAL_MOVE_ATTEMPTED"
    GAME_OVER = "GAME_OVER"


class GameStateManager:
    """
    Coordinates end-to-end physical game progression:
    Hand presence -> Stability check -> SSIM diff -> Legal move match -> Stockfish evaluation.
    """

    def __init__(
        self,
        engine_manager: EngineManager,
        calibrator: BoardCalibrator,
        hand_tracker: Optional[HandTracker] = None,
        voice_assistant: Optional[VoiceAssistant] = None,
        stability_frames_required: int = 6,
    ):
        self.engine = engine_manager
        self.calibrator = calibrator
        self.hand_tracker = hand_tracker or HandTracker()
        self.voice_assistant = voice_assistant
        self.detector = PhysicalBoardDetector(calibrator)

        self.stability_frames_required = stability_frames_required
        self.status = GameStatus.WAITING_FOR_MOVE
        
        self.ref_board_800: Optional[np.ndarray] = None
        self._prev_candidate_board_800: Optional[np.ndarray] = None
        self._stability_counter: int = 0
        self.last_move: Optional[chess.Move] = None

    def initialize_reference_board(self, initial_frame: np.ndarray) -> bool:
        """
        Sets the starting reference board image (at the beginning of game).
        Accepts either raw camera frames or pre-warped 800x800 boards.
        """
        if initial_frame.shape == (self.calibrator.target_size, self.calibrator.target_size, 3):
            warped = initial_frame.copy()
        else:
            warped = self.calibrator.warp_board(initial_frame)

        if warped is None:
            logger.error("Failed to initialize reference board: Calibration not set.")
            return False

        self.ref_board_800 = warped.copy()
        self.status = GameStatus.WAITING_FOR_MOVE
        logger.success("Reference board initialized successfully.")
        return True

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Processes a live video frame through the state machine.
        
        Returns:
            Dict containing status, hand_detected, detected_move, evaluation, and annotated_frame.
        """
        if self.ref_board_800 is None:
            self.initialize_reference_board(frame)

        # 1. Warp to 800x800 if not already in target dimensions
        if frame.shape == (self.calibrator.target_size, self.calibrator.target_size, 3):
            curr_board_800 = frame
            hand_present = self.hand_tracker.is_hand_over_board(frame, None)
        else:
            curr_board_800 = self.calibrator.warp_board(frame)
            if curr_board_800 is None:
                return {"status": self.status.value, "error": "Uncalibrated"}
            hand_present = self.hand_tracker.is_hand_over_board(frame, self.calibrator.corners)

        event = None
        move_info = None

        if hand_present:
            self.status = GameStatus.HAND_DETECTED
            self._stability_counter = 0
            self._prev_candidate_board_800 = None
        else:
            if self.status == GameStatus.HAND_DETECTED:
                # Hand just left, enter stabilizing phase
                self.status = GameStatus.STABILIZING
                self._stability_counter = 0
                self._prev_candidate_board_800 = curr_board_800.copy()

            elif self.status == GameStatus.STABILIZING:
                # Check pixel stability between consecutive frames without hand
                if self._prev_candidate_board_800 is not None:
                    diff = cv2.absdiff(curr_board_800, self._prev_candidate_board_800)
                    mean_motion = np.mean(diff)

                    if mean_motion < 8.0:  # Very little motion (stable)
                        self._stability_counter += 1
                    else:
                        self._stability_counter = 0  # Still settling

                    self._prev_candidate_board_800 = curr_board_800.copy()

                    # Once stabilized for required frames, trigger move detection
                    if self._stability_counter >= self.stability_frames_required:
                        detailed = self.detector.detect_move_detailed(
                            self.ref_board_800,
                            curr_board_800,
                            self.engine.board
                        )

                        if detailed["status"] == "LEGAL_MOVE":
                            move = detailed["move"]
                            conf = detailed["confidence"]
                            prev_board = self.engine.board.copy()
                            san = self.engine.board.san(move)
                            self.engine.make_move(move.uci())
                            self.last_move = move
                            self.ref_board_800 = curr_board_800.copy()
                            self.status = GameStatus.MOVE_CONFIRMED

                            # Trigger Stockfish response evaluation
                            analysis = self.engine.evaluate_position(time_limit=0.08)

                            # Trigger Persian Voice Announcements
                            if self.voice_assistant:
                                self.voice_assistant.announce_move(move, prev_board, is_opponent=True)
                                self.voice_assistant.announce_ai_recommendation(analysis)

                            # Check for terminal state
                            if self.engine.board.is_game_over():
                                self.status = GameStatus.GAME_OVER
                                if self.voice_assistant:
                                    self.voice_assistant.announce_game_over(self.engine.board.result())

                            event = "MOVE_DETECTED"
                            move_info = {
                                "move_uci": move.uci(),
                                "move_san": san,
                                "confidence": conf,
                                "engine_analysis": analysis,
                                "fen": self.engine.get_fen(),
                            }
                            logger.success(f"Physical move confirmed: {san} | Stockfish best reply: {analysis['best_move_san']}")

                        elif detailed["status"] == "ILLEGAL_MOVE":
                            self.status = GameStatus.ILLEGAL_MOVE_ATTEMPTED
                            event = "ILLEGAL_MOVE_DETECTED"
                            move_info = {
                                "attempted_uci": detailed["attempted_uci"],
                                "from_square": detailed["from_square"],
                                "to_square": detailed["to_square"],
                                "confidence": detailed["confidence"],
                            }
                            if self.voice_assistant:
                                self.voice_assistant.announce_threat("حرکت غیرقانونی است! لطفاً مهره را به جای خود بازگردانید.")
                            logger.warning(f"Illegal physical move rejected: {detailed['attempted_uci']}")

                        else:
                            # False alarm or minor bump: return to waiting
                            self.status = GameStatus.WAITING_FOR_MOVE

            elif self.status in [GameStatus.MOVE_CONFIRMED, GameStatus.ILLEGAL_MOVE_ATTEMPTED]:
                # Transition back to waiting for next move or piece replacement
                self.status = GameStatus.WAITING_FOR_MOVE

        return {
            "status": self.status.value,
            "hand_present": hand_present,
            "event": event,
            "move_info": move_info,
            "current_fen": self.engine.get_fen(),
        }
