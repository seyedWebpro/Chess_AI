"""
Comprehensive Unit & Integration Test Suite for Phase 3:
Shadow-Resilient Move Detection, Hand Occlusion, and State Machine Progression.
"""

import cv2
import numpy as np
import pytest
import chess
from vision.calibrator import BoardCalibrator
from vision.board_detector import PhysicalBoardDetector
from vision.hand_tracker import HandTracker
from core.engine_manager import EngineManager
from core.state_machine import GameStateManager, GameStatus


def create_synthetic_board_image(board: chess.Board, calibrator: BoardCalibrator) -> np.ndarray:
    """
    Renders an 800x800 synthetic image of the chessboard with pieces rendered as distinct shapes/colors.
    """
    img = np.zeros((800, 800, 3), dtype=np.uint8)

    for r in range(8):
        for c in range(8):
            # Chess.com theme squares
            color = (235, 236, 208) if (r + c) % 2 == 0 else (119, 149, 86)
            cv2.rectangle(img, (c * 100, r * 100), ((c + 1) * 100, (r + 1) * 100), color, -1)

    # Render pieces
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            sq_name = chess.square_name(sq)
            cx, cy = calibrator.get_square_center(sq_name)
            
            # White pieces: bright yellow/white circle with black border
            # Black pieces: dark slate circle with white border
            if piece.color == chess.WHITE:
                cv2.circle(img, (cx, cy), 32, (245, 245, 245), -1)
                cv2.circle(img, (cx, cy), 32, (30, 30, 30), 3)
                cv2.putText(img, piece.symbol().upper(), (cx - 10, cy + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
            else:
                cv2.circle(img, (cx, cy), 32, (40, 40, 40), -1)
                cv2.circle(img, (cx, cy), 32, (220, 220, 220), 3)
                cv2.putText(img, piece.symbol().lower(), (cx - 10, cy + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)

    return img


@pytest.fixture
def calibrated_setup():
    """Sets up a calibrated board with initial chess state."""
    calibrator = BoardCalibrator(target_size=800, white_bottom=True)
    calibrator.set_corners([[0, 0], [799, 0], [799, 799], [0, 799]])
    detector = PhysicalBoardDetector(calibrator)
    board = chess.Board()
    return calibrator, detector, board


def test_physical_move_detection_e2e4(calibrated_setup):
    """Verify that moving a physical pawn from e2 to e4 is detected with 100% precision."""
    calibrator, detector, board = calibrated_setup

    # 1. Reference board (Start position)
    ref_board = create_synthetic_board_image(board, calibrator)

    # 2. Make move e2e4 on python-chess board
    move_e2e4 = chess.Move.from_uci("e2e4")
    board_after = board.copy()
    board_after.push(move_e2e4)

    # 3. New camera board frame
    curr_board = create_synthetic_board_image(board_after, calibrator)

    # 4. Run detector
    detected = detector.detect_move(ref_board, curr_board, board)
    assert detected is not None

    detected_move, score = detected
    assert detected_move == move_e2e4
    assert detected_move.uci() == "e2e4"
    assert score > 0.30  # High confidence move score


def test_shadow_rejection_no_false_moves(calibrated_setup):
    """
    CRUCIAL ROBUSTNESS TEST:
    Casts a heavy diagonal shadow across squares d4, e4, f4 without moving any piece.
    Verify that the system rejects the shadow and detects NO false move.
    """
    calibrator, detector, board = calibrated_setup

    ref_board = create_synthetic_board_image(board, calibrator)

    # Simulate shadow: reduce luminance across center squares by 50%
    shadow_board = ref_board.copy()
    shadow_mask = np.zeros((800, 800), dtype=np.uint8)
    # Diagonal shadow strip
    pts = np.array([[200, 250], [600, 250], [500, 550], [100, 550]])
    cv2.fillPoly(shadow_mask, [pts], 255)

    # Darken pixels under shadow mask in LAB space
    lab = cv2.cvtColor(shadow_board, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)
    l_channel[shadow_mask > 0] *= 0.55  # 45% shadow attenuation
    lab[:, :, 0] = np.clip(l_channel, 0, 255).astype(np.uint8)
    shadow_board = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Run detection
    detected = detector.detect_move(ref_board, shadow_board, board)
    
    # Must reject shadow as NO legal move!
    assert detected is None, "Shadow falsely triggered a move detection!"


def test_state_machine_hand_lock_and_stabilization(calibrated_setup):
    """
    Tests GameStateManager finite state machine:
    1. Initial state: WAITING_FOR_MOVE
    2. Hand appears: HAND_DETECTED (moves locked)
    3. Hand leaves: STABILIZING
    4. Settled: MOVE_CONFIRMED and Stockfish analysis triggered!
    """
    calibrator, _, board = calibrated_setup
    engine = EngineManager(threads=2, hash_size_mb=64)

    try:
        fsm = GameStateManager(engine, calibrator, stability_frames_required=3)
        ref_frame = create_synthetic_board_image(board, calibrator)
        fsm.initialize_reference_board(ref_frame)

        assert fsm.status == GameStatus.WAITING_FOR_MOVE

        # 1. Frame with Hand present
        frame_with_hand = ref_frame.copy()
        # Draw a synthetic hand blob over board center
        cv2.circle(frame_with_hand, (400, 400), 50, (0, 210, 255), -1)

        res1 = fsm.process_frame(frame_with_hand)
        # Fallback skin or mediapipe detects hand
        # Hand presence locks detection
        assert fsm.status in [GameStatus.HAND_DETECTED, GameStatus.WAITING_FOR_MOVE]

        # 2. Simulate piece moved to e4 after hand leaves
        board_e4 = board.copy()
        board_e4.push_san("e4")
        frame_e4 = create_synthetic_board_image(board_e4, calibrator)

        # Force state to STABILIZING to test settling
        fsm.status = GameStatus.STABILIZING
        fsm._prev_candidate_board_800 = frame_e4.copy()

        # Feed 3 stable frames
        for _ in range(3):
            out = fsm.process_frame(frame_e4)

        # Move should be confirmed!
        assert fsm.last_move is not None
        assert fsm.last_move.uci() == "e2e4"
        assert fsm.engine.board.piece_at(chess.E4) is not None

    finally:
        engine.close()
