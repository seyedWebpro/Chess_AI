"""
Unit and Integration Tests for ARDrawer (Phase 4: AR Visual Overlay & Video Pipeline).
"""

import time
import pytest
import numpy as np
import cv2
from ui.ar_drawer import ARDrawer
from vision.calibrator import BoardCalibrator


@pytest.fixture
def sample_calibrator() -> BoardCalibrator:
    calib = BoardCalibrator(target_size=800, white_bottom=True)
    # Tilted chessboard corners in 1280x720 frame
    physical_corners = np.array([
        [280, 110],   # a8
        [1000, 130],  # h8
        [1160, 660],  # h1
        [140, 630]    # a1
    ], dtype=np.float32)
    calib.set_corners(physical_corners)
    return calib


@pytest.fixture
def sample_camera_frame() -> np.ndarray:
    return np.full((720, 1280, 3), 40, dtype=np.uint8)


@pytest.fixture
def sample_warped_board() -> np.ndarray:
    board = np.zeros((800, 800, 3), dtype=np.uint8)
    for r in range(8):
        for c in range(8):
            col = (235, 236, 208) if (r + c) % 2 == 0 else (119, 149, 86)
            cv2.rectangle(board, (c * 100, r * 100), ((c + 1) * 100, (r + 1) * 100), col, -1)
    return board


def test_01_ar_drawer_initialization():
    drawer = ARDrawer(enable_glow=True)
    assert drawer.enable_glow is True
    assert drawer.COLOR_NEON_GREEN == (94, 197, 34)
    assert drawer.COLOR_AMBER_YELLOW == (8, 179, 234)
    assert drawer.COLOR_CRIMSON_RED == (68, 68, 239)

    drawer_no_glow = ARDrawer(enable_glow=False)
    assert drawer_no_glow.enable_glow is False


def test_02_draw_smooth_arrow(sample_camera_frame):
    drawer = ARDrawer(enable_glow=True)
    frame = sample_camera_frame.copy()
    start_pt = (200, 200)
    end_pt = (400, 400)

    res = drawer.draw_smooth_arrow(frame, start_pt, end_pt, color=drawer.COLOR_NEON_GREEN, thickness=6)

    assert res.shape == (720, 1280, 3)
    assert res.dtype == np.uint8
    # Pixels near start and end should not be original background (40)
    assert not np.all(res[200, 200] == 40)
    assert not np.all(res[350, 350] == 40)


def test_03_draw_square_polygon(sample_camera_frame):
    drawer = ARDrawer()
    frame = sample_camera_frame.copy()
    quad = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32)

    res = drawer.draw_square_polygon(frame, quad, color=drawer.COLOR_AMBER_YELLOW, alpha=0.4, border=True)

    assert res.shape == (720, 1280, 3)
    # Center of square should have blended color
    center_color = res[150, 150]
    assert not np.all(center_color == 40)


def test_04_draw_eval_bar(sample_camera_frame):
    drawer = ARDrawer()

    # White winning (80%)
    frame_white_lead = sample_camera_frame.copy()
    res_w = drawer.draw_eval_bar(frame_white_lead, win_chance_pct=80.0, eval_str="+3.5", x=20, y=50, w=25, h=400)
    assert res_w.shape == (720, 1280, 3)
    # The lower part of the bar should be white
    lower_bar_pixel = res_w[400, 32]
    assert np.all(lower_bar_pixel == 245)

    # Black winning (20%)
    frame_black_lead = sample_camera_frame.copy()
    res_b = drawer.draw_eval_bar(frame_black_lead, win_chance_pct=20.0, eval_str="-3.5", x=20, y=50, w=25, h=400)
    # The upper part should be dark (40)
    upper_bar_pixel = res_b[100, 32]
    assert np.all(upper_bar_pixel == 40)


def test_05_draw_hud(sample_camera_frame):
    drawer = ARDrawer()

    # Test winning HUD (Green stripe)
    analysis_winning = {
        "best_move": "e2e4",
        "best_move_san": "e4",
        "eval_str": "+2.4",
        "win_chance_pct": 72.0,
        "depth_reached": 18,
    }
    frame = sample_camera_frame.copy()
    res = drawer.draw_hud(frame, analysis_winning, x=50, y=50, w=350, h=60)
    # Check stripe color at x+2, y+10
    stripe_px = res[60, 52]
    # In BGR, neon green is (94, 197, 34)
    assert stripe_px[1] > 150  # High green channel

    # Test losing HUD (Red stripe)
    analysis_losing = {
        "best_move": "e7e5",
        "best_move_san": "e5",
        "eval_str": "-2.5",
        "win_chance_pct": 30.0,
        "depth_reached": 16,
    }
    frame_losing = sample_camera_frame.copy()
    res_losing = drawer.draw_hud(frame_losing, analysis_losing, x=50, y=50, w=350, h=60)
    stripe_losing_px = res_losing[60, 52]
    # In BGR, crimson red is (68, 68, 239) -> high R channel (index 2)
    assert stripe_losing_px[2] > 200


def test_06_render_ar_on_camera(sample_camera_frame, sample_calibrator):
    drawer = ARDrawer(enable_glow=True)
    frame = sample_camera_frame.copy()

    analysis = {
        "best_move": "e2e4",
        "best_move_san": "e4",
        "from_square": "e2",
        "to_square": "e4",
        "eval_str": "+1.8",
        "win_chance_pct": 68.0,
        "depth_reached": 20,
    }

    result = drawer.render_ar_on_camera(
        camera_frame=frame,
        analysis=analysis,
        calibrator=sample_calibrator,
        check_square="e1",
    )

    assert result.shape == frame.shape
    assert result.dtype == np.uint8
    # Ensure overlay is drawn and not identical to blank frame
    assert not np.array_equal(result, frame)


def test_07_render_ar_on_warped(sample_warped_board, sample_calibrator):
    drawer = ARDrawer()
    analysis = {
        "best_move": "g1f3",
        "best_move_san": "Nf3",
        "from_square": "g1",
        "to_square": "f3",
        "eval_str": "+0.4",
        "win_chance_pct": 53.0,
        "depth_reached": 16,
    }

    result = drawer.render_ar_on_warped(sample_warped_board, analysis, sample_calibrator)
    assert result.shape == (800, 800, 3)
    assert not np.array_equal(result, sample_warped_board)


def test_08_rendering_performance_benchmark(sample_camera_frame, sample_calibrator):
    drawer = ARDrawer(enable_glow=True)
    analysis = {
        "best_move": "d2d4",
        "best_move_san": "d4",
        "from_square": "d2",
        "to_square": "d4",
        "eval_str": "+0.5",
        "win_chance_pct": 54.0,
        "depth_reached": 18,
    }

    # Warmup
    for _ in range(5):
        _ = drawer.render_ar_on_camera(sample_camera_frame, analysis, sample_calibrator, check_square="e1")

    # Benchmark 60 iterations
    iterations = 60
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = drawer.render_ar_on_camera(sample_camera_frame, analysis, sample_calibrator, check_square="e1")
    total_time_ms = (time.perf_counter() - t0) * 1000
    avg_latency_ms = total_time_ms / iterations

    print(f"\n[BENCHMARK] AR Camera Rendering Latency: {avg_latency_ms:.3f} ms / frame ({1000/avg_latency_ms:.1f} FPS)")
    # Must render well within 60 FPS budget (16.6ms), target sub-10ms (100+ FPS)
    assert avg_latency_ms < 10.0, f"AR rendering took {avg_latency_ms:.2f}ms, exceeded 10ms limit!"
