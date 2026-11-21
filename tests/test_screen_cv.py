"""
Unit and Integration Tests for Screen CV Module (Phase 6: Online Chess Screen Capture & CV).
"""

import time
from typing import Optional, Dict
import pytest
import numpy as np
import cv2
import chess

from screen.screen_capture import ScreenCapture, ThreadedScreenCapture, MockScreenCapture
from screen.screen_board_detector import ScreenBoardDetector
from screen.screen_move_detector import ScreenMoveDetector


def create_synthetic_screen_with_board(
    board_x: int = 350,
    board_y: int = 120,
    board_size: int = 640,
    highlight_squares: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    """
    Creates a full 1440x900 desktop screen containing browser chrome, UI, and a Chess.com board.
    """
    screen = np.full((900, 1440, 3), 30, dtype=np.uint8)  # dark desktop wallpaper
    # Browser window
    cv2.rectangle(screen, (200, 50), (1240, 850), (45, 45, 45), -1)
    # Browser top bar
    cv2.rectangle(screen, (200, 50), (1240, 95), (60, 60, 60), -1)

    # Chess.com Board
    sq_size = board_size // 8
    for r in range(8):
        for c in range(8):
            # Chess.com standard green theme BGR:
            # Light: (208, 236, 235), Dark: (86, 149, 119)
            color = (208, 236, 235) if (r + c) % 2 == 0 else (86, 149, 119)
            x1 = board_x + c * sq_size
            y1 = board_y + r * sq_size
            cv2.rectangle(screen, (x1, y1), (x1 + sq_size, y1 + sq_size), color, -1)

    # Add border around board
    cv2.rectangle(screen, (board_x - 2, board_y - 2), (board_x + board_size + 2, board_y + board_size + 2), (15, 15, 15), 2)

    # Draw initial pieces
    temp_board = chess.Board()
    for sq in chess.SQUARES:
        piece = temp_board.piece_at(sq)
        if piece:
            col = chess.square_file(sq)
            row = 7 - chess.square_rank(sq)
            cx = board_x + col * sq_size + sq_size // 2
            cy = board_y + row * sq_size + sq_size // 2
            p_col = (245, 245, 245) if piece.color == chess.WHITE else (30, 30, 30)
            cv2.circle(screen, (cx, cy), sq_size // 3, p_col, -1)

    # Apply highlights if any
    if highlight_squares:
        for sq_name in highlight_squares:
            col = ord(sq_name[0]) - ord('a')
            row = 7 - (int(sq_name[1]) - 1)
            x1 = board_x + col * sq_size
            y1 = board_y + row * sq_size
            # Chess.com yellow highlight BGR: (130, 246, 245) for light, (67, 202, 185) for dark
            hl_col = (130, 246, 245) if (row + col) % 2 == 0 else (67, 202, 185)
            # Fill square with highlight
            cv2.rectangle(screen, (x1, y1), (x1 + sq_size, y1 + sq_size), hl_col, -1)
            # Re-draw piece if present
            sq_idx = chess.parse_square(sq_name)
            p = temp_board.piece_at(sq_idx)
            if p:
                cx = x1 + sq_size // 2
                cy = y1 + sq_size // 2
                p_col = (245, 245, 245) if p.color == chess.WHITE else (30, 30, 30)
                cv2.circle(screen, (cx, cy), sq_size // 3, p_col, -1)

    return screen


def test_01_screen_capture_init_and_mock():
    mock = MockScreenCapture()
    frame = np.full((600, 600, 3), 128, dtype=np.uint8)
    mock.set_frame(frame)
    grabbed = mock.grab()
    assert grabbed.shape == (600, 600, 3)

    sc = ScreenCapture()
    sc.set_roi(10, 10, 200, 200)
    assert sc.bbox["width"] == 200
    sc.clear_roi()
    assert sc.bbox is None
    sc.close()


def test_02_threaded_screen_capture():
    mock_sc = ScreenCapture()
    threaded = ThreadedScreenCapture(target_fps=20)
    threaded.start()
    time.sleep(0.1)
    frame = threaded.read()
    if frame is not None:
        assert len(frame.shape) == 3
    threaded.stop()


def test_03_board_auto_localization():
    # Synthetic desktop with Chess.com board at (350, 120, 640, 640)
    desktop = create_synthetic_screen_with_board(board_x=350, board_y=120, board_size=640)
    detector = ScreenBoardDetector()
    bbox = detector.find_board_auto(desktop)

    assert bbox is not None
    x, y, w, h = bbox
    # Must localize accurately within 10 pixels
    assert abs(x - 350) < 10
    assert abs(y - 120) < 10
    assert abs(w - 640) < 10
    assert abs(h - 640) < 10


def test_04_orientation_detection():
    detector = ScreenBoardDetector()
    desktop = create_synthetic_screen_with_board(board_x=100, board_y=100, board_size=640)
    board_crop = desktop[100:740, 100:740]

    # In our synthetic board White is at bottom (standard)
    is_white_bottom = detector.detect_orientation(board_crop)
    assert is_white_bottom is True

    # Invert (Black at bottom)
    board_flipped = cv2.rotate(board_crop, cv2.ROTATE_180)
    is_white_bottom_flipped = detector.detect_orientation(board_flipped)
    assert is_white_bottom_flipped is False


def test_05_chess_com_highlight_move_detection():
    # Simulate Chess.com move e2 -> e4:
    # Squares e2 and e4 highlighted in yellow
    desktop = create_synthetic_screen_with_board(
        board_x=200,
        board_y=100,
        board_size=640,
        highlight_squares={"e2": "yellow", "e4": "yellow"},
    )
    board_crop = desktop[100:740, 200:840]

    board_det = ScreenBoardDetector(board_bbox=(200, 100, 640, 640))
    move_det = ScreenMoveDetector(board_detector=board_det)

    game_board = chess.Board()
    detected = move_det.detect_move_from_highlights(board_crop, game_board)

    assert detected is not None
    move, conf = detected
    assert move.uci() == "e2e4"
    assert conf > 0.60


def test_06_knight_move_highlight_detection():
    # Simulate move g1 -> f3
    game_board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    desktop = create_synthetic_screen_with_board(
        board_x=200,
        board_y=100,
        board_size=640,
        highlight_squares={"g1": "yellow", "f3": "yellow"},
    )
    board_crop = desktop[100:740, 200:840]

    board_det = ScreenBoardDetector(board_bbox=(200, 100, 640, 640))
    move_det = ScreenMoveDetector(board_detector=board_det)

    detected = move_det.detect_move_from_highlights(board_crop, game_board)
    assert detected is not None
    move, conf = detected
    assert move.uci() == "g1f3"
    assert conf > 0.60


def test_07_highlight_detection_latency_benchmark():
    # Benchmark 100 iterations of move highlight detection
    desktop = create_synthetic_screen_with_board(
        board_x=200,
        board_y=100,
        board_size=640,
        highlight_squares={"e2": "yellow", "e4": "yellow"},
    )
    board_crop = desktop[100:740, 200:840]

    board_det = ScreenBoardDetector(board_bbox=(200, 100, 640, 640))
    move_det = ScreenMoveDetector(board_detector=board_det)
    game_board = chess.Board()

    # Warmup
    for _ in range(5):
        move_det.last_confirmed_move = None
        _ = move_det.detect_move_from_highlights(board_crop, game_board)

    iterations = 100
    t0 = time.perf_counter()
    for _ in range(iterations):
        move_det.last_confirmed_move = None
        _ = move_det.detect_move_from_highlights(board_crop, game_board)
    total_time_ms = (time.perf_counter() - t0) * 1000
    avg_latency_ms = total_time_ms / iterations

    print(f"\n[BENCHMARK] Screen Move Highlight Detection Latency: {avg_latency_ms:.3f} ms / frame ({1000/avg_latency_ms:.0f} FPS)")
    # Must be sub-2.0 milliseconds
    assert avg_latency_ms < 2.0, f"Highlight detection took {avg_latency_ms:.2f}ms, exceeded 2ms target!"


def test_08_retrigger_protection():
    desktop = create_synthetic_screen_with_board(
        board_x=200,
        board_y=100,
        board_size=640,
        highlight_squares={"e2": "yellow", "e4": "yellow"},
    )
    board_crop = desktop[100:740, 200:840]

    board_det = ScreenBoardDetector(board_bbox=(200, 100, 640, 640))
    move_det = ScreenMoveDetector(board_detector=board_det)
    game_board = chess.Board()

    # First detection -> confirms e2e4
    res1 = move_det.detect_move_from_highlights(board_crop, game_board)
    assert res1 is not None

    # Immediate second detection on SAME screen -> returns None (re-trigger protection)
    res2 = move_det.detect_move_from_highlights(board_crop, game_board)
    assert res2 is None
