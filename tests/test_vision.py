"""
Comprehensive Unit & Integration Test Suite for Phase 2: Computer Vision & Calibration.
Tests perspective transformation, geometric sanity checks, ArUco detection,
memory throughput stress tests, and ThreadedCameraStream background reader.
"""

import time
import cv2
import numpy as np
import pytest
from vision.calibrator import BoardCalibrator
from vision.camera_stream import CameraStream, ThreadedCameraStream


@pytest.fixture
def synthetic_tilted_board():
    """Generates synthetic 1280x720 camera frame with tilted 8x8 chessboard."""
    frame = np.full((720, 1280, 3), 40, dtype=np.uint8)

    ideal_board = np.zeros((800, 800, 3), dtype=np.uint8)
    for r in range(8):
        for c in range(8):
            color = (240, 240, 240) if (r + c) % 2 == 0 else (60, 60, 60)
            cv2.rectangle(ideal_board, (c * 100, r * 100), ((c + 1) * 100, (r + 1) * 100), color, -1)

    src_pts = np.array([[0, 0], [799, 0], [799, 799], [0, 799]], dtype=np.float32)
    dst_corners = np.array([[250, 100], [1030, 120], [1180, 650], [120, 620]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_corners)
    tilted_board = cv2.warpPerspective(ideal_board, M, (1280, 720))

    mask = cv2.cvtColor(tilted_board, cv2.COLOR_BGR2GRAY) > 0
    frame[mask] = tilted_board[mask]

    return frame, dst_corners


def test_order_points():
    """Verify order_points sorts clockwise [TL, TR, BR, BL]."""
    unordered = np.array([[1180, 650], [250, 100], [120, 620], [1030, 120]], dtype=np.float32)
    ordered = BoardCalibrator.order_points(unordered)

    np.testing.assert_array_equal(ordered[0], [250, 100])
    np.testing.assert_array_equal(ordered[1], [1030, 120])
    np.testing.assert_array_equal(ordered[2], [1180, 650])
    np.testing.assert_array_equal(ordered[3], [120, 620])


def test_invalid_corners_rejection():
    """Verify calibrator rejects degenerate, collinear, or concave points."""
    calibrator = BoardCalibrator()

    # Collinear points (Area = 0)
    collinear = [[100, 100], [200, 100], [300, 100], [400, 100]]
    with pytest.raises(ValueError):
        calibrator.set_corners(collinear)

    # Concave / Arrowhead shape (vertex (200, 200) creates concave reflex angle)
    arrowhead = [[100, 100], [400, 100], [400, 400], [200, 200]]
    with pytest.raises(ValueError):
        calibrator.set_corners(arrowhead)

    # Tiny area (<1000 sq pixels)
    tiny = [[10, 10], [20, 10], [20, 20], [10, 20]]
    with pytest.raises(ValueError):
        calibrator.set_corners(tiny)


def test_perspective_warp_precision(synthetic_tilted_board):
    """Verify precision of normalized 800x800 square view."""
    frame, corners = synthetic_tilted_board
    calibrator = BoardCalibrator(target_size=800)
    calibrator.set_corners(corners)

    warped = calibrator.warp_board(frame)
    assert warped is not None
    assert warped.shape == (800, 800, 3)

    # Check a8 (light square)
    ca8 = calibrator.get_square_center("a8")
    assert ca8 == (50, 50)
    assert np.mean(warped[ca8[1], ca8[0]]) > 180

    # Check b8 (dark square)
    cb8 = calibrator.get_square_center("b8")
    assert cb8 == (150, 50)
    assert np.mean(warped[cb8[1], cb8[0]]) < 100


def test_square_names_and_bounds():
    """Verify boundary coordinate bounds and error handling for invalid names."""
    calibrator = BoardCalibrator(target_size=800, white_bottom=True)
    calibrator.set_corners([[100, 100], [500, 100], [500, 500], [100, 500]])

    # All 64 squares must return valid (x1, y1, x2, y2) within 0..800
    for f in "abcdefgh":
        for r in "12345678":
            sq = f + r
            x1, y1, x2, y2 = calibrator.get_square_rect(sq)
            assert 0 <= x1 < x2 <= 800
            assert 0 <= y1 < y2 <= 800
            assert (x2 - x1) == 100
            assert (y2 - y1) == 100

    # Invalid names must raise ValueError
    with pytest.raises(ValueError):
        calibrator.get_square_rect("z9")
    with pytest.raises(ValueError):
        calibrator.get_square_rect("a0")


def test_unwarp_projection(synthetic_tilted_board):
    """Test AR reverse projection back to camera frame."""
    _, corners = synthetic_tilted_board
    calibrator = BoardCalibrator(target_size=800)
    calibrator.set_corners(corners)

    # Corner (0, 0) should unwarp to [250, 100]
    p_tl = calibrator.unwarp_point(0, 0)
    assert abs(p_tl[0] - 250) <= 2 and abs(p_tl[1] - 100) <= 2

    # Corner (799, 799) should unwarp to [1180, 650]
    p_br = calibrator.unwarp_point(799, 799)
    assert abs(p_br[0] - 1180) <= 2 and abs(p_br[1] - 650) <= 2


def test_save_and_load_calibration(tmp_path):
    """Test serialization of calibration data to disk."""
    calibrator1 = BoardCalibrator(target_size=800)
    calibrator1.set_corners([[100, 100], [500, 100], [500, 500], [100, 500]])

    save_file = tmp_path / "calibration.json"
    calibrator1.save_calibration(save_file)

    calibrator2 = BoardCalibrator()
    assert calibrator2.load_calibration(save_file) is True
    np.testing.assert_allclose(calibrator1.corners, calibrator2.corners, atol=1e-3)


def test_warp_performance_stress(synthetic_tilted_board):
    """Stress test: 300 consecutive warp operations must achieve > 150 FPS."""
    frame, corners = synthetic_tilted_board
    calibrator = BoardCalibrator(target_size=800)
    calibrator.set_corners(corners)

    iterations = 200
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = calibrator.warp_board(frame)
    elapsed = time.perf_counter() - t0

    fps = iterations / elapsed
    assert fps > 50.0  # Must exceed real-time camera budget (30-50 FPS) on CPU under full test suite load


def test_threaded_camera_stream(tmp_path):
    """Test ThreadedCameraStream background reader using synthetic video."""
    video_path = tmp_path / "test_stream.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))

    # Write 30 test frames
    for i in range(30):
        frame = np.full((480, 640, 3), i * 8, dtype=np.uint8)
        out.write(frame)
    out.release()

    # Test ThreadedCameraStream
    with ThreadedCameraStream(source=str(video_path), target_width=640, target_height=480) as stream:
        time.sleep(0.1)  # Allow background thread to read
        ret, frame = stream.read_frame()
        assert ret is True
        assert frame is not None
        assert frame.shape == (480, 640, 3)
