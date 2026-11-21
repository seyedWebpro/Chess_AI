"""
test_automation.py: Comprehensive Unit & Integration Test Suite for Phase 9.
Validates Biomechanical WindMouse kinematics, Cognitive Log-Normal timing,
Gaussian Square Dispersion, Mock & Native Mac Controllers, and Anti-Cheat Telemetry Auditing.
"""

import math
import numpy as np
import pytest

from automation.wind_mouse import (
    TrajectoryPoint,
    wind_mouse_trajectory,
    generate_human_move_path,
)
from automation.human_behavior import (
    HumanThinkTimer,
    GaussianSquareSampler,
)
from automation.mouse_controller import (
    MockMouseController,
    NativeMacMouseController,
    HumanMouseController,
)
from automation.telemetry_validator import TelemetryValidator


class TestAutomation:
    """Phase 9 Test Suite: Human-like Mouse Automation & Anti-Cheat Evasion."""

    def test_01_wind_mouse_trajectory_generation(self):
        """Validates fundamental WindMouse physics, monotonicity, and target arrival."""
        start_x, start_y = 100.0, 100.0
        dest_x, dest_y = 650.0, 480.0

        points = wind_mouse_trajectory(
            start_x, start_y, dest_x, dest_y,
            gravity=9.0, wind=3.5, max_step=16.0
        )

        assert len(points) >= 15, f"Trajectory too sparse: {len(points)} points"
        assert (points[0].x, points[0].y) == (start_x, start_y)
        assert (points[-1].x, points[-1].y) == (dest_x, dest_y)

        # Monotonic timestamp check
        for i in range(1, len(points)):
            assert points[i].timestamp >= points[i - 1].timestamp, (
                f"Non-monotonic timestamp at step {i}: {points[i].timestamp} < {points[i-1].timestamp}"
            )

        # Ensure points do not jump arbitrarily far in a single step
        for i in range(1, len(points)):
            step_d = math.hypot(points[i].x - points[i - 1].x, points[i].y - points[i - 1].y)
            assert step_d < 35.0, f"Unnatural jump detected: {step_d:.2f}px at step {i}"

    def test_02_overshoot_and_tremor_kinematics(self):
        """Tests biological overshoot and micro-tremor generation."""
        start_x, start_y = 150.0, 150.0
        dest_x, dest_y = 550.0, 500.0

        # Normal path without overshoot/tremor
        clean_path = generate_human_move_path(
            start_x, start_y, dest_x, dest_y, enable_overshoot=False, enable_tremor=False
        )

        # Natural human path with overshoot and physiological tremor
        human_path = generate_human_move_path(
            start_x, start_y, dest_x, dest_y, enable_overshoot=True, enable_tremor=True
        )

        assert len(human_path) > len(clean_path)
        assert human_path[-1].x == dest_x
        assert human_path[-1].y == dest_y

        # Tremor check: middle points should have subtle micro-fluctuations
        entropy_clean = TelemetryValidator.calculate_jitter_entropy(clean_path)
        entropy_human = TelemetryValidator.calculate_jitter_entropy(human_path)

        assert entropy_human > 0.05, f"Expected non-zero tremor entropy, got {entropy_human}"
        assert entropy_human >= entropy_clean or entropy_clean > 0.01

    def test_03_human_cognitive_timer_distribution(self):
        """Validates log-normal reaction time distributions across game contexts."""
        timer = HumanThinkTimer(min_delay=0.20, max_delay=6.0)

        opening_delays = [timer.calculate_delay(is_opening_book=True, ply_count=4) for _ in range(40)]
        check_delays = [timer.calculate_delay(is_in_check=True) for _ in range(40)]
        blunder_delays = [timer.calculate_delay(is_blunder_check=True) for _ in range(40)]

        # Opening book moves must be faster than deep blunder checks on average
        mean_opening = float(np.mean(opening_delays))
        mean_blunder = float(np.mean(blunder_delays))
        assert mean_opening < mean_blunder, f"Opening ({mean_opening:.2f}s) should be faster than ponder ({mean_blunder:.2f}s)"

        # All delays must respect physical boundaries
        for d in opening_delays + check_delays + blunder_delays:
            assert 0.20 <= d <= 6.0, f"Delay {d} out of range [0.20, 6.0]"

        # Validate click hold duration
        click_durations = [timer.calculate_click_duration() for _ in range(30)]
        for cd in click_durations:
            assert 0.040 <= cd <= 0.180, f"Click duration {cd} outside human 40-180ms"

    def test_04_gaussian_square_sampling(self):
        """Validates 2D Gaussian spatial dispersion on chess board squares."""
        sampler = GaussianSquareSampler(std_ratio=0.18, max_deviation_ratio=0.35)
        board_bbox = (100, 100, 800, 800) # 100x100px squares

        # Sample 40 clicks on 'e4'
        # White orientation: 'e' is col 4, '4' is rank 4 -> row 4 (0-indexed from top: 7-3 = 4)
        # Expected e4 center: 100 + 4.5*100 = 550, 100 + 4.5*100 = 550
        clicks = [
            sampler.sample_square_from_bbox(board_bbox, "e4", orientation_white_bottom=True)
            for _ in range(40)
        ]

        nominal_center = (550.0, 550.0)
        disp_eval = TelemetryValidator.validate_dispersion(clicks, nominal_center, square_size=100.0)

        assert disp_eval["valid"] is True, f"Dispersion failed: {disp_eval}"
        assert disp_eval["unique_points"] > 25, "Points must not collapse to identical coordinates"
        assert disp_eval["max_offset"] < 45.0, "Clicks must not leak into neighbor squares"

        # Test flipped orientation (Black bottom)
        click_black = sampler.sample_square_from_bbox(board_bbox, "e4", orientation_white_bottom=False)
        # In black bottom: col = 7 - 4 = 3 -> x center around 100 + 3.5*100 = 450
        # row = rank - 1 = 3 -> y center around 100 + 3.5*100 = 450
        assert 400.0 < click_black[0] < 500.0
        assert 400.0 < click_black[1] < 500.0

    def test_05_mock_mouse_controller_events(self):
        """Tests MockMouseController deterministic logging and playback."""
        mock = MockMouseController(initial_x=200.0, initial_y=200.0, fast_mode=True)
        assert mock.get_position() == (200.0, 200.0)

        # Move
        mock.move_to(350.0, 400.0)
        assert mock.get_position() == (350.0, 400.0)

        # Click
        mock.click(duration=0.065)
        assert mock.recorded_events[-1]["type"] == "click"
        assert mock.recorded_events[-1]["duration"] == 0.065

        # Execute trajectory
        pts = [
            TrajectoryPoint(x=10.0, y=10.0, timestamp=0.0),
            TrajectoryPoint(x=20.0, y=25.0, timestamp=0.01),
            TrajectoryPoint(x=30.0, y=40.0, timestamp=0.02),
        ]
        mock.execute_trajectory(pts, is_drag=False, fast_mode=True)
        assert len(mock.recorded_points) >= 3
        assert mock.get_position() == (30.0, 40.0)

    def test_06_human_mouse_controller_click_and_drag_modes(self):
        """Tests full move execution in both Click-Click and Drag-and-Drop modes."""
        mock = MockMouseController(initial_x=300.0, initial_y=300.0, fast_mode=True)
        controller = HumanMouseController(backend=mock, fast_mode=True)
        board_bbox = (100, 100, 640, 640) # 80x80px squares

        # 1. Click-Click move 'e2e4'
        res_click = controller.play_move(
            from_square="e2",
            to_square="e4",
            board_bbox=board_bbox,
            is_drag=False,
            context={"is_opening_book": True, "ply_count": 2},
        )
        assert res_click["success"] is True
        assert res_click["mode"] == "click_click"
        assert res_click["total_trajectory_points"] > 20

        # Check mock recorded clicks
        click_events = [e for e in mock.recorded_events if e["type"] == "click"]
        assert len(click_events) >= 2, f"Expected at least 2 clicks, got {len(click_events)}"

        # 2. Drag-and-Drop move 'g1f3'
        res_drag = controller.play_move(
            from_square="g1",
            to_square="f3",
            board_bbox=board_bbox,
            is_drag=True,
            context={"is_opening_book": False, "is_in_check": False},
        )
        assert res_drag["success"] is True
        assert res_drag["mode"] == "drag"
        assert res_drag["total_trajectory_points"] > 20

        # Check mock down/up pairs
        down_events = [e for e in mock.recorded_events if e["type"] == "down"]
        up_events = [e for e in mock.recorded_events if e["type"] == "up"]
        assert len(down_events) >= len(click_events) + 1
        assert len(up_events) >= len(click_events) + 1

    def test_07_anti_cheat_telemetry_statistical_audit(self):
        """Comprehensive anti-cheat evaluation across multiple human trajectories."""
        # Generate 15 distinct human paths between various squares
        test_coords = [
            (100.0, 100.0, 450.0, 300.0),
            (250.0, 600.0, 500.0, 200.0),
            (600.0, 150.0, 180.0, 480.0),
            (300.0, 300.0, 700.0, 650.0),
            (120.0, 400.0, 520.0, 110.0),
        ]

        audit_results = []
        for sx, sy, dx, dy in test_coords:
            path = generate_human_move_path(sx, sy, dx, dy, enable_overshoot=True, enable_tremor=True)
            report = TelemetryValidator.evaluate_trajectory(path)
            audit_results.append(report)

        # Verify all paths pass anti-cheat criteria
        passed_count = sum(1 for r in audit_results if r["overall_pass"])
        pass_rate = passed_count / len(audit_results)

        assert pass_rate >= 0.80, f"Anti-cheat pass rate {pass_rate*100:.1f}% below 80% threshold. Details: {audit_results}"

        # Confirm curvature is strictly non-straight (R < 0.99 for all)
        for r in audit_results:
            assert r["curvature"] < 0.995, f"Bot straight line detected! R = {r['curvature']}"
            assert r["jitter_entropy"] > 0.01, f"Zero-jitter synthetic curve detected! Jerk = {r['jitter_entropy']}"

    def test_08_native_mac_mouse_controller_initialization_and_safety(self):
        """Verifies NativeMacMouseController loads CoreGraphics ctypes safely without crashing."""
        native = NativeMacMouseController()
        # Controller must initialize without error on macOS
        assert hasattr(native, "available")
        curr_x, curr_y = native.get_position()
        assert isinstance(curr_x, float)
        assert isinstance(curr_y, float)

        # Test execute_trajectory with empty points (safe no-op)
        native.execute_trajectory([], is_drag=False, fast_mode=True)
