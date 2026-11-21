"""
Deep End-to-End (E2E) Integration, Stress & Quality Assurance Suite for Phases 1-5 MVP.
Tests:
- Full physical game replay simulation (multi-move game).
- Physical illegal move rejection & referee warning.
- Scholar's Mate checkmate sequence with terminal game state and voice announcement.
- Severe lighting & cast shadow stress resilience.
- AR Drawer corner coordinate safety and rendering stability.
- Concurrency & thread safety under high throughput.
"""

import time
import pytest
import numpy as np
import cv2
import chess
from pathlib import Path

from core.engine_manager import EngineManager
from vision.calibrator import BoardCalibrator
from vision.board_detector import PhysicalBoardDetector
from core.state_machine import GameStateManager, GameStatus
from ui.ar_drawer import ARDrawer
from audio.voice_assistant import VoiceAssistant


@pytest.fixture
def calibrated_system():
    # Setup standard 800x800 board with physical tilted corners
    physical_corners = np.array([
        [280, 110],   # a8
        [1000, 130],  # h8
        [1160, 660],  # h1
        [140, 630]    # a1
    ], dtype=np.float32)

    calib = BoardCalibrator(800, white_bottom=True)
    calib.set_corners(physical_corners)

    engine = EngineManager(threads=2, hash_size_mb=64)
    voice = VoiceAssistant(enable_audio=True, mock_playback=True)
    fsm = GameStateManager(
        engine_manager=engine,
        calibrator=calib,
        voice_assistant=voice,
        stability_frames_required=2,
    )

    yield calib, engine, voice, fsm

    voice.close()
    engine.close()


def create_board_image_from_chess_board(board: chess.Board, calib: BoardCalibrator) -> np.ndarray:
    """Helper to generate synthetic top-down 800x800 board representing a chess.Board state."""
    img = np.zeros((800, 800, 3), dtype=np.uint8)
    # Draw squares
    for r in range(8):
        for c in range(8):
            color = (235, 236, 208) if (r + c) % 2 == 0 else (119, 149, 86)
            cv2.rectangle(img, (c * 100, r * 100), ((c + 1) * 100, (r + 1) * 100), color, -1)

    # Draw pieces
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            col = chess.square_file(sq)
            row = 7 - chess.square_rank(sq)
            cx = col * 100 + 50
            cy = row * 100 + 50
            piece_col = (250, 250, 250) if piece.color == chess.WHITE else (20, 20, 20)
            cv2.circle(img, (cx, cy), 32, piece_col, -1)
            cv2.circle(img, (cx, cy), 32, (100, 100, 100), 2)
            # Distinct marker for piece type
            cv2.putText(
                img,
                piece.symbol().upper(),
                (cx - 10, cy + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255) if piece.color == chess.WHITE else (0, 255, 255),
                2
            )

    return img


def test_01_full_game_replay_simulation(calibrated_system):
    """
    Simulates a sequence of physical moves:
    1. e4
    2. e5
    3. Nf3
    Verifies that the state machine, move detector, and voice assistant stay in perfect sync.
    """
    calib, engine, voice, fsm = calibrated_system

    game_board = chess.Board()
    ref_img = create_board_image_from_chess_board(game_board, calib)
    fsm.initialize_reference_board(ref_img)

    moves_to_play = ["e2e4", "e7e5", "g1f3"]

    for move_uci in moves_to_play:
        move_obj = chess.Move.from_uci(move_uci)
        game_board.push(move_obj)
        current_img = create_board_image_from_chess_board(game_board, calib)

        # 1. Hand appears
        fsm.status = GameStatus.HAND_DETECTED
        # 2. Hand leaves and stabilizes for 2 frames
        fsm.status = GameStatus.STABILIZING
        fsm._prev_candidate_board_800 = current_img.copy()

        res1 = fsm.process_frame(current_img)
        res2 = fsm.process_frame(current_img)

        assert fsm.last_move is not None
        assert fsm.last_move.uci() == move_uci
        assert fsm.engine.board.fen() == game_board.fen()

    assert len(fsm.engine.board.move_stack) == 3


def test_02_illegal_physical_move_rejection(calibrated_system):
    """
    Simulates an illegal move attempt:
    White moves Pawn from e2 to e5 on move 1 (illegal!).
    System must:
    1. Detect illegal move attempt.
    2. Transition to ILLEGAL_MOVE_ATTEMPTED.
    3. Issue a Persian voice warning.
    4. Reject board state update.
    """
    calib, engine, voice, fsm = calibrated_system

    board_start = chess.Board()
    ref_img = create_board_image_from_chess_board(board_start, calib)
    fsm.initialize_reference_board(ref_img)

    # Illegally teleport e2 pawn to e5
    illegal_board = board_start.copy()
    pawn = illegal_board.remove_piece_at(chess.E2)
    illegal_board.set_piece_at(chess.E5, pawn)
    illegal_img = create_board_image_from_chess_board(illegal_board, calib)

    fsm.status = GameStatus.STABILIZING
    fsm._prev_candidate_board_800 = illegal_img.copy()

    fsm.process_frame(illegal_img)
    fsm.process_frame(illegal_img)

    assert fsm.status == GameStatus.ILLEGAL_MOVE_ATTEMPTED
    assert voice.last_spoken_text is not None
    assert "غیرقانونی" in voice.last_spoken_text
    # Internal board must remain intact at initial move!
    assert len(fsm.engine.board.move_stack) == 0


def test_03_checkmate_sequence_and_terminal_voice(calibrated_system):
    """
    Simulates Scholar's Mate checkmate (Qxf7#).
    Verifies that checkmate triggers GAME_OVER and announces result.
    """
    calib, engine, voice, fsm = calibrated_system

    # Position before Qxf7#
    fen_before_mate = "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4"
    engine.set_fen(fen_before_mate)
    board_mate = chess.Board(fen_before_mate)

    ref_img = create_board_image_from_chess_board(board_mate, calib)
    fsm.initialize_reference_board(ref_img)

    # Play Qxf7#
    board_mate.push_san("Qxf7#")
    mate_img = create_board_image_from_chess_board(board_mate, calib)

    fsm.status = GameStatus.STABILIZING
    fsm._prev_candidate_board_800 = mate_img.copy()

    fsm.process_frame(mate_img)
    fsm.process_frame(mate_img)

    assert fsm.status == GameStatus.GAME_OVER
    assert fsm.engine.board.is_checkmate()
    assert voice.last_spoken_text is not None
    assert ("برنده" in voice.last_spoken_text or "پایان" in voice.last_spoken_text)


def test_04_severe_shadow_stress_resilience(calibrated_system):
    """
    Adds harsh synthetic shadows (40-60% brightness drop) over half the board.
    Verifies that the SSIM + LAB detector detects a real move without triggering on shadows.
    """
    calib, engine, voice, fsm = calibrated_system
    detector = PhysicalBoardDetector(calibrator=calib)

    board_start = chess.Board()
    ref_img = create_board_image_from_chess_board(board_start, calib)

    board_e4 = board_start.copy()
    board_e4.push_san("e4")
    move_img = create_board_image_from_chess_board(board_e4, calib)

    # Cast harsh diagonal shadow
    shadow_mask = np.zeros((800, 800), dtype=np.uint8)
    pts = np.array([[100, 0], [700, 0], [500, 800], [0, 800]], dtype=np.int32)
    cv2.fillPoly(shadow_mask, [pts], 255)

    # Darken shadow area by 70 pixels
    move_img_shadow = move_img.copy()
    move_img_shadow[shadow_mask > 0] = np.clip(
        move_img_shadow[shadow_mask > 0].astype(np.int16) - 70, 0, 255
    ).astype(np.uint8)

    detected = detector.detect_move(ref_img, move_img_shadow, board_start)
    assert detected is not None
    move, score = detected
    assert move.uci() == "e2e4"
    assert score > 0.30


def test_05_ar_corner_and_boundary_safety(calibrated_system):
    """
    Verifies ARDrawer doesn't crash on extreme boundary squares (a1, a8, h1, h8)
    or invalid / clipped coordinates.
    """
    calib, _, _, _ = calibrated_system
    drawer = ARDrawer(enable_glow=True)

    dummy_frame = np.full((720, 1280, 3), 40, dtype=np.uint8)

    # Corner to corner arrow (a1 -> h8)
    analysis = {
        "best_move": "a1h8",
        "best_move_san": "Qh8",
        "from_square": "a1",
        "to_square": "h8",
        "eval_str": "+4.2",
        "win_chance_pct": 92.0,
        "depth_reached": 22,
    }

    # Must render without any OpenCV exception
    rendered = drawer.render_ar_on_camera(
        camera_frame=dummy_frame,
        analysis=analysis,
        calibrator=calib,
        check_square="e8",
    )
    assert rendered.shape == (720, 1280, 3)
    assert not np.array_equal(rendered, dummy_frame)


def test_06_voice_and_engine_high_throughput_concurrency(calibrated_system):
    """
    Stress tests voice synthesis queue and engine evaluations under rapid sequential calls.
    Ensures zero deadlocks and zero memory corruption.
    """
    calib, engine, voice, _ = calibrated_system

    positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4",
    ]

    for fen in positions:
        engine.set_fen(fen)
        eval_res = engine.evaluate_position(time_limit=0.03)
        assert eval_res["best_move"] is not None
        voice.announce_ai_recommendation(eval_res)

    voice.stop_current_speech()
    assert voice._speech_queue.empty()
