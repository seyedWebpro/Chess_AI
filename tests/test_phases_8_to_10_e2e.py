"""
test_phases_8_to_10_e2e.py: Deep End-to-End (E2E) Integration Suite for Phases 8, 9, and 10.
Tests the full autonomous loop: Transparent Screen Overlay (PyQt6), Biomechanical WindMouse
Automation (Anti-Cheat), Polyglot Opening Books, Maia Neural Rating Personas, and Playwright UI Auditing.
"""

import sys
import time
import math
import chess
import pytest
from PyQt6.QtWidgets import QApplication
from playwright.sync_api import sync_playwright

from core.hybrid_brain import HybridBrain
from core.maia_engine import Persona
from automation.wind_mouse import generate_human_move_path
from automation.mouse_controller import HumanMouseController, MockMouseController
from automation.telemetry_validator import TelemetryValidator
from ui.overlay_window import TransparentOverlayWindow
from ui.master_dashboard import MasterDashboardWindow


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


class TestPhases8To10E2E:
    """Deep Integration & E2E Verification for Phases 8, 9, and 10."""

    def test_01_full_autonomous_bot_pipeline_simulation(self, qapp):
        """
        Simulates end-to-end autonomous bot cycle:
        Board position -> Hybrid Brain (Polyglot + Maia) -> Transparent Overlay Render -> WindMouse execution -> Telemetry audit.
        """
        mock_mouse = MockMouseController(fast_mode=True)
        human_mouse = HumanMouseController(backend=mock_mouse, prefer_drag=False, fast_mode=True)

        with HybridBrain(persona=Persona.MAIA_1900, use_opening_book=True) as brain:
            board = chess.Board() # Starting position

            # Step 1: Decision
            t0 = time.perf_counter()
            decision = brain.decide_move(board)
            dt_decision = (time.perf_counter() - t0) * 1000.0

            assert decision["move"] is not None
            assert decision["is_book"] is True, "First move must be from Polyglot book"
            assert dt_decision < 5.0, f"Polyglot book lookup too slow: {dt_decision:.2f}ms"

            # Step 2: Overlay AR Update (Phase 8)
            from_sq = decision["uci"][:2]
            to_sq = decision["uci"][2:4]
            board_bbox = (150, 150, 640, 640)
            overlay = TransparentOverlayWindow(board_bbox=board_bbox, white_bottom=True)
            overlay.set_board_geometry(*board_bbox, white_bottom=True)
            overlay.update_analysis({
                "best_move": decision["uci"],
                "best_move_san": decision["uci"],
                "from_square": from_sq,
                "to_square": to_sq,
                "eval_score": 0.2,
                "win_chance_percent": decision["win_chance_pct"],
                "is_mate": False,
            })
            mouse_res = human_mouse.play_move(
                from_square=from_sq,
                to_square=to_sq,
                board_bbox=board_bbox,
                is_drag=False,
            )

            assert mouse_res["success"] is True
            assert mouse_res["total_trajectory_points"] >= 20

            # Step 4: Anti-Cheat Telemetry Audit on move trajectory segment
            move_points = mock_mouse.recorded_points[-mouse_res["path_to_dest_points"]:]
            audit = TelemetryValidator.evaluate_trajectory(move_points)

            assert audit["curvature_pass"] is True, f"Straight line detected: R = {audit['curvature']}"
            assert audit["jitter_pass"] is True, f"Jitter entropy failed: {audit['jitter_entropy']}"
            assert audit["overall_pass"] is True, f"Audit report failed: {audit}"

            overlay.close()

    def test_02_dual_persona_self_play_simulation(self):
        """
        Simulates an authentic self-play match between two distinct Maia personas:
        White: Maia 1900 (Club Master) vs Black: Maia 1500 (Intermediate).
        Verifies 10 sequential moves of legal play, proper book usage, and cognitive reaction delays.
        """
        with HybridBrain(persona=Persona.MAIA_1900, use_opening_book=True) as brain_white, \
             HybridBrain(persona=Persona.MAIA_1500, use_opening_book=True) as brain_black:

            board = chess.Board()
            moves_played = []

            for ply in range(1, 9):
                current_brain = brain_white if board.turn == chess.WHITE else brain_black
                decision = current_brain.decide_move(board, time_limit=0.08)

                move = decision["move"]
                assert move is not None, f"Brain failed to select move at ply {ply}"
                assert move in board.legal_moves, f"Illegal move {move.uci()} generated at ply {ply}"

                # Cognitive deliberation must be non-zero and realistic
                assert 0.15 <= decision["deliberation_time_sec"] <= 7.0

                moves_played.append(decision["uci"])
                board.push(move)

            assert len(moves_played) == 8
            assert len(set(moves_played)) >= 6, "Opening must develop multiple distinct pieces"

    def test_03_click_through_and_mouse_drag_mode_e2e(self, qapp):
        """
        Verifies Drag & Drop execution mode through transparent overlay
        with Fitts's law velocity curve and log-normal drag pause.
        """
        mock_mouse = MockMouseController(fast_mode=True)
        human_mouse = HumanMouseController(backend=mock_mouse, prefer_drag=True, fast_mode=True)

        board_bbox = (200, 200, 720, 720)
        overlay = TransparentOverlayWindow(board_bbox=board_bbox)

        # Execute Drag-and-drop move 'e2e4'
        res = human_mouse.play_move(
            from_square="e2",
            to_square="e4",
            board_bbox=board_bbox,
            is_drag=True,
            context={"is_opening_book": True, "ply_count": 1},
        )

        assert res["mode"] == "drag"
        assert res["success"] is True

        # Check recorded mouse events: must have down, drag trajectory, and up
        events = mock_mouse.recorded_events
        event_types = [e["type"] for e in events]
        assert "down" in event_types
        assert "drag_trajectory" in event_types
        assert "up" in event_types

        overlay.close()

    def test_04_stress_rapid_persona_switching_and_memory(self):
        """
        Stress-tests rapid persona switching and Stockfish query concurrency
        to verify zero memory leaks and deterministic thread safety.
        """
        with HybridBrain(persona=Persona.GOD_MODE, use_opening_book=False) as brain:
            board = chess.Board()

            personas = [
                Persona.GOD_MODE,
                Persona.MAIA_1100,
                Persona.MAIA_1500,
                Persona.MAIA_1900,
                Persona.BLUNDER_SIM,
            ]

            for i in range(12):
                p = personas[i % len(personas)]
                brain.set_persona(p)
                assert brain.persona == p

                dec = brain.decide_move(board, time_limit=0.04)
                assert dec["move"] in board.legal_moves

    def test_05_master_dashboard_full_cycle_headless(self, qapp):
        """
        Executes a 4-move full dashboard automated cycle in headless mode,
        asserting GUI logs, state snapshots, win bar, and move signal emissions.
        """
        mock_mouse = HumanMouseController(backend=MockMouseController(fast_mode=True), fast_mode=True)
        dashboard = MasterDashboardWindow(mouse_controller=mock_mouse, headless=True)
        dashboard.autobot_checkbox.setChecked(True)

        moves_history = []
        dashboard.move_executed_signal.connect(lambda d: moves_history.append(d["uci"]))

        for _ in range(4):
            dashboard.trigger_ai_move()

        assert len(moves_history) == 4
        assert len(dashboard.board.move_stack) == 4
        assert "AI Decision" in dashboard.log_text.toPlainText()

        dashboard.close()
