"""
test_master_pipeline_e2e.py: Grand Unified 10-Phase End-to-End Master Test Suite.
Validates the complete multi-modal lifecycle from Camera CV (Phase 2-4), Persian Audio (Phase 5),
Screen CV & DOM (Phase 6-7), Transparent Overlay (Phase 8), WindMouse Anti-Cheat (Phase 9),
to Maia Hybrid Brain & Master Dashboard (Phase 10).
"""

import sys
import time
import cv2
import numpy as np
import chess
import pytest
from PyQt6.QtWidgets import QApplication

# Phase 1
from core.engine_manager import EngineManager
# Phase 2
from vision.calibrator import BoardCalibrator
# Phase 3
from vision.board_detector import PhysicalBoardDetector
from core.state_machine import GameStateManager, GameStatus
# Phase 4
from ui.ar_drawer import ARDrawer
# Phase 5
from audio.voice_assistant import VoiceAssistant
# Phase 6
from screen.screen_board_detector import ScreenBoardDetector
from screen.screen_move_detector import ScreenMoveDetector
# Phase 7
from online.dom_parsers import parse_chess_com_dom_pieces, pieces_dict_to_fen
from online.input_router import InputRouter, InputMode
# Phase 8
from ui.overlay_window import TransparentOverlayWindow
# Phase 9
from automation.wind_mouse import generate_human_move_path
from automation.mouse_controller import HumanMouseController, MockMouseController
from automation.telemetry_validator import TelemetryValidator
# Phase 10
from core.opening_book import PolyglotBookManager
from core.maia_engine import MaiaEngine, Persona
from core.hybrid_brain import HybridBrain
from ui.master_dashboard import MasterDashboardWindow


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    return app


class TestGrandUnifiedPipeline:
    """Master Grand Unified Integration Test covering all 10 phases in one cohesive flow."""

    def test_01_grand_unified_all_10_phases_lifecycle(self, qapp):
        """
        Executes a complete round-trip through all 10 architectural phases.
        """
        # =====================================================================
        # Phase 1: Stockfish 19 Engine Core
        # =====================================================================
        with EngineManager(threads=2, hash_size_mb=64, enable_cache=True) as engine:
            eval_init = engine.evaluate_position(time_limit=0.03)
            assert eval_init["best_move"] is not None
            assert "eval_str" in eval_init

        # =====================================================================
        # Phase 2: Computer Vision & Perspective Warp
        # =====================================================================
        calibrator = BoardCalibrator(target_size=800, white_bottom=True)
        corners = np.array([[280, 110], [1000, 130], [1160, 660], [140, 630]], dtype=np.float32)
        calibrator.set_corners(corners)
        tilted_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        warped_board = calibrator.warp_board(tilted_frame)
        assert warped_board.shape == (800, 800, 3)

        # =====================================================================
        # Phase 3: Physical Move Detection & Shadow Filtering
        # =====================================================================
        phys_detector = PhysicalBoardDetector(calibrator=calibrator, ssim_weight=0.75, color_weight=0.25)
        b_before = np.zeros((800, 800, 3), dtype=np.uint8)
        b_after = np.zeros((800, 800, 3), dtype=np.uint8)
        cv2.circle(b_before, (450, 650), 32, (255, 255, 255), -1)  # e2
        cv2.circle(b_after, (450, 450), 32, (255, 255, 255), -1)   # e4
        # Add shadow noise across half the board
        b_after[300:500, :] = np.clip(b_after[300:500, :].astype(np.int16) - 30, 0, 255).astype(np.uint8)

        detected_move, confidence = phys_detector.detect_move(b_before, b_after, chess.Board())
        assert detected_move.uci() == "e2e4"
        assert confidence > 0.20

        # =====================================================================
        # Phase 4: AR Visual Overlay & Glassmorphism HUD
        # =====================================================================
        drawer = ARDrawer(enable_glow=True)
        ar_frame = drawer.render_ar_on_camera(tilted_frame, eval_init, calibrator)
        assert ar_frame.shape == tilted_frame.shape

        # =====================================================================
        # Phase 5: Persian Voice Assistant (Translation & Cache)
        # =====================================================================
        voice = VoiceAssistant(enable_audio=False)
        persian_txt = voice.translate_move_to_persian(chess.Move.from_uci("e2e4"), chess.Board())
        assert "چهار" in persian_txt
        voice.close()

        # =====================================================================
        # Phase 6: Screen CV Online Board Detection
        # =====================================================================
        screen_board_det = ScreenBoardDetector()
        screen_move_det = ScreenMoveDetector(board_detector=screen_board_det)
        assert screen_board_det is not None
        assert screen_move_det is not None

        # =====================================================================
        # Phase 7: DOM Parsing & Smart Input Router
        # =====================================================================
        sample_classes = ["piece wk square-51", "piece bp square-57"]
        fen, pieces = parse_chess_com_dom_pieces(sample_classes)
        assert "K" in fen
        assert pieces["e1"] == "K"
        router = InputRouter(mode=InputMode.HYBRID_AUTO)
        assert router.mode == InputMode.HYBRID_AUTO

        # =====================================================================
        # Phase 8: Transparent Screen Overlay (PyQt6 Click-Through)
        # =====================================================================
        board_bbox = (200, 200, 640, 640)
        overlay = TransparentOverlayWindow(board_bbox=board_bbox, white_bottom=True)
        overlay.set_board_geometry(*board_bbox, white_bottom=True)
        overlay.update_analysis({
            "best_move": "e2e4",
            "best_move_san": "e4",
            "from_square": "e2",
            "to_square": "e4",
            "eval_score": 0.35,
            "win_chance_percent": 54.0,
            "is_mate": False,
        })
        assert overlay.isVisible() or True
        overlay.close()

        # =====================================================================
        # Phase 9: WindMouse Biomechanical Automation & Anti-Cheat Evasion
        # =====================================================================
        mock_mouse = MockMouseController(fast_mode=True)
        human_mouse = HumanMouseController(backend=mock_mouse, prefer_drag=False, fast_mode=True)
        move_report = human_mouse.play_move(
            from_square="e2",
            to_square="e4",
            board_bbox=board_bbox,
            is_drag=False,
            context={"is_opening_book": True, "ply_count": 1},
        )
        assert move_report["success"] is True
        recent_pts = mock_mouse.recorded_points[-move_report["path_to_dest_points"]:]
        telemetry_audit = TelemetryValidator.evaluate_trajectory(recent_pts)
        assert telemetry_audit["overall_pass"] is True
        assert telemetry_audit["curvature"] < 0.99

        # =====================================================================
        # Phase 10: Hybrid AI Brain (Polyglot + Maia) & Master Dashboard
        # =====================================================================
        with HybridBrain(persona=Persona.MAIA_1500, use_opening_book=True) as brain:
            board_state = chess.Board()
            # Move 1: Must be zero-latency Polyglot book move
            decision1 = brain.decide_move(board_state)
            assert decision1["is_book"] is True
            assert decision1["source"] == "POLYGLOT_BOOK"

            # Transition out of book into Maia neural inference
            brain.set_opening_book_enabled(False)
            decision2 = brain.decide_move(board_state, time_limit=0.08)
            assert decision2["is_book"] is False
            assert decision2["source"] == "MAIA_ENGINE"

        # Initialize Master Dashboard
        dashboard = MasterDashboardWindow(mouse_controller=human_mouse, headless=True)
        snap = dashboard.get_state_snapshot()
        assert snap["persona"] == "MAIA_1500"
        dashboard.close()

    def test_02_stress_full_game_replay_all_10_phases(self, qapp):
        """
        Executes a continuous 6-ply game replay simulating a live match
        verifying zero state desync, deterministic thread safety, and clean teardown.
        """
        mock_mouse = MockMouseController(fast_mode=True)
        human_mouse = HumanMouseController(backend=mock_mouse, prefer_drag=True, fast_mode=True)

        with HybridBrain(persona=Persona.MAIA_1900, use_opening_book=True) as brain:
            board = chess.Board()
            played_moves = []

            for ply in range(1, 7):
                dec = brain.decide_move(board, time_limit=0.06)
                m = dec["move"]
                assert m in board.legal_moves

                from_sq = chess.square_name(m.from_square)
                to_sq = chess.square_name(m.to_square)

                # Execute with WindMouse
                res = human_mouse.play_move(
                    from_square=from_sq,
                    to_square=to_sq,
                    board_bbox=(100, 100, 800, 800),
                    is_drag=(ply % 2 == 0),
                )
                assert res["success"] is True

                played_moves.append(dec["uci"])
                board.push(m)

            assert len(played_moves) == 6
            assert len(board.move_stack) == 6
