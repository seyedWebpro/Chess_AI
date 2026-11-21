"""
tests/test_overlay_window.py: Unit and Performance Tests for
Phase 8: Transparent & Click-Through Screen Overlay (PyQt6).
"""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import time
from pathlib import Path
import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QColor, QImage

from ui.overlay_window import TransparentOverlayWindow


@pytest.fixture(scope="session")
def qapp():
    """Provides singleton QApplication instance for test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_01_overlay_initialization_and_flags(qapp):
    win = TransparentOverlayWindow(board_bbox=(200, 100, 640, 640))

    # Assert Window Flags
    flags = win.windowFlags()
    assert bool(flags & Qt.WindowType.FramelessWindowHint)
    assert bool(flags & Qt.WindowType.WindowStaysOnTopHint)

    # Assert Transparency & Click-Through Attributes
    assert win.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    assert win.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    assert win.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    assert win.click_through is True


def test_02_square_coordinate_math(qapp):
    # Board at x=200, y=100, size=640x640 -> square size = 80px
    win = TransparentOverlayWindow(board_bbox=(200, 100, 640, 640), white_bottom=True)

    # a1: bottom-left square -> col=0, row=7
    # center: x = 200 + 0*80 + 40 = 240, y = 100 + 7*80 + 40 = 700
    p_a1 = win.square_to_center_point("a1")
    assert p_a1 is not None
    assert p_a1.x() == 240.0
    assert p_a1.y() == 700.0

    # h8: top-right square -> col=7, row=0
    # center: x = 200 + 7*80 + 40 = 800, y = 100 + 0*80 + 40 = 140
    p_h8 = win.square_to_center_point("h8")
    assert p_h8 is not None
    assert p_h8.x() == 800.0
    assert p_h8.y() == 140.0

    # e4: col=4 ('e'), rank=4 ('4' -> row=7-3=4)
    # center: x = 200 + 4*80 + 40 = 560, y = 100 + 4*80 + 40 = 460
    p_e4 = win.square_to_center_point("e4")
    assert p_e4 is not None
    assert p_e4.x() == 560.0
    assert p_e4.y() == 460.0

    # Square bounds
    rect_e4 = win.square_to_rect("e4")
    assert rect_e4 is not None
    assert rect_e4.x() == 520.0
    assert rect_e4.y() == 420.0
    assert rect_e4.width() == 80.0
    assert rect_e4.height() == 80.0


def test_03_flipped_orientation_coordinates(qapp):
    # Board with Black at bottom (flipped)
    win = TransparentOverlayWindow(board_bbox=(200, 100, 640, 640), white_bottom=False)

    # When flipped: a1 is top-right (col=7, row=0)
    p_a1_flipped = win.square_to_center_point("a1")
    assert p_a1_flipped is not None
    assert p_a1_flipped.x() == 800.0
    assert p_a1_flipped.y() == 140.0

    # h8 is bottom-left (col=0, row=7)
    p_h8_flipped = win.square_to_center_point("h8")
    assert p_h8_flipped is not None
    assert p_h8_flipped.x() == 240.0
    assert p_h8_flipped.y() == 700.0


def test_04_click_through_toggle(qapp):
    win = TransparentOverlayWindow(board_bbox=(200, 100, 640, 640))
    assert win.click_through is True

    # Disable click-through (e.g. interactive calibration mode)
    win.set_click_through(False)
    assert win.click_through is False
    assert not win.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    # Re-enable click-through
    win.set_click_through(True)
    assert win.click_through is True
    assert win.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)


def test_05_update_and_clear_analysis(qapp):
    win = TransparentOverlayWindow(board_bbox=(200, 100, 640, 640))

    analysis = {
        "best_move": "f3f7",
        "best_move_san": "Qxf7#",
        "from_square": "f3",
        "to_square": "f7",
        "eval_score": 99.0,
        "is_mate": True,
        "mate_in": 1,
        "win_chance_percent": 100.0,
        "threat_squares": ["e8", "d8"],
    }
    win.update_analysis(analysis)

    assert win.best_move == "f3f7"
    assert win.best_move_san == "Qxf7#"
    assert win.from_square == "f3"
    assert win.to_square == "f7"
    assert win.is_mate is True
    assert win.mate_in == 1
    assert "e8" in win.threat_squares

    # Clear overlay
    win.clear_overlay()
    assert win.best_move is None
    assert win.best_move_san is None
    assert len(win.threat_squares) == 0


def test_06_offscreen_render_and_pixel_verification(qapp):
    win = TransparentOverlayWindow(board_bbox=(200, 100, 640, 640))
    win.update_analysis({
        "best_move": "e2e4",
        "best_move_san": "e4",
        "from_square": "e2",
        "to_square": "e4",
        "eval_score": 1.5,
        "win_chance_percent": 65.0,
        "threat_squares": ["f7"],
    })

    pixmap = win.render_to_pixmap(width=1280, height=900)
    assert not pixmap.isNull()
    assert pixmap.width() == 1280
    assert pixmap.height() == 900

    # Convert to QImage and verify rendered non-transparent pixels exist
    img = pixmap.toImage()
    # Check that pixel at arrow center or HUD has non-zero alpha
    hud_center_pixel = img.pixelColor(200 + 320, 100 - 37)
    assert hud_center_pixel.alpha() > 100, "HUD badge should be rendered with solid alpha."

    # Save fixture image for report
    fixtures_dir = Path(__file__).parent / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    save_path = fixtures_dir / "10_phase8_pyqt6_overlay_render.png"
    pixmap.save(str(save_path), "PNG")
    assert save_path.exists()


def test_07_rendering_performance_benchmark(qapp):
    win = TransparentOverlayWindow(board_bbox=(200, 100, 640, 640))
    win.update_analysis({
        "best_move": "e2e4",
        "best_move_san": "e4",
        "from_square": "e2",
        "to_square": "e4",
        "eval_score": 1.5,
        "win_chance_percent": 65.0,
        "threat_squares": ["f7"],
    })

    # Warmup
    for _ in range(5):
        _ = win.render_to_pixmap(width=1280, height=900)

    iterations = 100
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = win.render_to_pixmap(width=1280, height=900)
    total_time_ms = (time.perf_counter() - t0) * 1000
    avg_latency_ms = total_time_ms / iterations

    print(f"\n[BENCHMARK] PyQt6 Transparent Overlay Render: {avg_latency_ms:.3f} ms / frame ({1000/avg_latency_ms:.0f} FPS)")
    # Must easily achieve 60 FPS budget (< 16.6ms), targeting high performance
    assert avg_latency_ms < 12.0, f"Overlay render latency {avg_latency_ms:.2f}ms exceeded budget!"


def test_08_dynamic_board_bbox_update(qapp):
    win = TransparentOverlayWindow(board_bbox=(100, 50, 400, 400))
    p1 = win.square_to_center_point("e4")

    # Update geometry to new board position (e.g. user moved browser window)
    win.set_board_geometry(400, 200, 800, 800, white_bottom=True)
    p2 = win.square_to_center_point("e4")

    assert p1 is not None and p2 is not None
    assert p1.x() != p2.x()
    assert p1.y() != p2.y()
    # For 800x800 board at (400, 200): sq_size=100. e4 is col=4, row=4 -> cx=400+450=850, cy=200+450=650
    assert p2.x() == 850.0
    assert p2.y() == 650.0
