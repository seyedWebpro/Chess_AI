"""
test_hybrid_brain.py: Comprehensive Unit & Integration Test Suite for Phase 10.
Validates Polyglot Opening Book, Maia Rating Personas, Boltzmann Softmax Policy Distributions,
Hybrid Brain Seamless Transitions, and Master Dashboard PyQt6 GUI Controls.
"""

import sys
import chess
import pytest
from PyQt6.QtWidgets import QApplication

from core.opening_book import PolyglotBookManager, build_polyglot_binary_file
from core.maia_engine import MaiaEngine, Persona, PERSONA_PROFILES
from core.hybrid_brain import HybridBrain
from ui.master_dashboard import MasterDashboardWindow
from automation.mouse_controller import HumanMouseController, MockMouseController


@pytest.fixture(scope="module")
def qapp():
    """Provides or creates a single QApplication instance for Qt tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


class TestPhase10HybridBrain:
    """Phase 10 Test Suite: Hybrid AI Brain, Maia Personas, and Master Dashboard."""

    def test_01_polyglot_opening_book_lookup_and_weights(self):
        """Validates Polyglot binary opening book queries, Zobrist hashes, and weights."""
        with PolyglotBookManager() as book:
            board = chess.Board()
            assert book.is_in_book(board) is True

            all_moves = book.get_all_book_moves(board)
            assert len(all_moves) >= 3, f"Expected opening candidate moves, got {len(all_moves)}"

            ucis = [m[0].uci() for m in all_moves]
            assert "e2e4" in ucis
            assert "d2d4" in ucis

            # Query response to 1. e4
            board.push_san("e4")
            black_moves = book.get_all_book_moves(board)
            assert len(black_moves) >= 3
            black_ucis = [m[0].uci() for m in black_moves]
            assert "e7e5" in black_ucis or "c7c5" in black_ucis

            # Top move test
            top_move = book.get_book_move(board, mode="top")
            assert top_move in board.legal_moves

    def test_02_polyglot_out_of_book_transition(self):
        """Verifies that non-standard moves gracefully return out of book (None)."""
        with PolyglotBookManager() as book:
            board = chess.Board()
            # Play weird non-theoretical move sequence: 1. h4 h5 2. a4 a5
            board.push_san("h4")
            board.push_san("h5")
            board.push_san("a4")
            board.push_san("a5")

            assert book.is_in_book(board) is False
            assert book.get_book_move(board) is None

    def test_03_maia_engine_persona_switching(self):
        """Tests Maia rating persona dynamic configuration and profile parameters."""
        with MaiaEngine(persona=Persona.MAIA_1100) as maia:
            assert maia.persona == Persona.MAIA_1100
            assert PERSONA_PROFILES[maia.persona]["target_elo"] == 1100

            maia.set_persona(Persona.MAIA_1900)
            assert maia.persona == Persona.MAIA_1900
            assert PERSONA_PROFILES[maia.persona]["target_elo"] == 1900

            maia.set_persona(Persona.GOD_MODE)
            assert maia.persona == Persona.GOD_MODE
            assert PERSONA_PROFILES[maia.persona]["temperature"] == 0.001

    def test_04_maia_boltzmann_softmax_distribution(self):
        """Validates Boltzmann softmax probability distribution calculation."""
        with MaiaEngine(persona=Persona.MAIA_1500) as maia:
            board = chess.Board()
            pred = maia.predict_move(board, time_limit=0.10)

            assert pred["move"] in board.legal_moves
            assert pred["persona"] == "MAIA_1500"
            assert isinstance(pred["probabilities"], dict)

            # Probabilities must sum to 1.0 within tolerance
            total_prob = sum(pred["probabilities"].values())
            assert abs(total_prob - 1.0) < 0.02, f"Probabilities do not sum to 1.0: {total_prob}"

    def test_05_hybrid_brain_book_to_engine_seamless_transition(self):
        """Verifies seamless automatic transition from opening book to engine calculation."""
        with HybridBrain(persona=Persona.MAIA_1500, use_opening_book=True) as brain:
            board = chess.Board()

            # 1. Starting position -> should come from Polyglot book
            res1 = brain.decide_move(board)
            assert res1["is_book"] is True
            assert res1["source"] == "POLYGLOT_BOOK"
            assert res1["deliberation_time_sec"] > 0.0

            # 2. Force out of book by disabling book
            brain.set_opening_book_enabled(False)
            res2 = brain.decide_move(board, time_limit=0.10)
            assert res2["is_book"] is False
            assert res2["source"] == "MAIA_ENGINE"
            assert res2["move"] in board.legal_moves

    def test_06_hybrid_brain_god_mode_mate_tactics(self):
        """Verifies that God Mode finds checkmate with 100% precision."""
        with HybridBrain(persona=Persona.GOD_MODE, use_opening_book=False) as brain:
            # Scholar's mate puzzle position: 1. e4 e5 2. Bc4 Nc6 3. Qf3 Nf6?? 4. Qxf7#
            puzzle_fen = "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4"
            board = chess.Board(puzzle_fen)

            res = brain.decide_move(board, time_limit=0.15)
            assert res["source"] == "STOCKFISH_GOD_MODE"
            assert res["uci"] == "f3f7", f"God Mode must play Qxf7#, got {res['uci']}"
            assert res["win_chance_pct"] == 100.0

    def test_07_master_dashboard_initialization_and_ui_bindings(self, qapp):
        """Verifies MasterDashboardWindow initialization and programmatic snapshot API."""
        mock_mouse = HumanMouseController(backend=MockMouseController(fast_mode=True), fast_mode=True)
        dashboard = MasterDashboardWindow(mouse_controller=mock_mouse, headless=True)

        snapshot = dashboard.get_state_snapshot()
        assert "fen" in snapshot
        assert snapshot["persona"] == "MAIA_1500"
        assert snapshot["opening_book_enabled"] is True
        assert snapshot["autobot_enabled"] is False

        # Switch persona via combo
        dashboard.persona_combo.setCurrentIndex(0) # God Mode
        assert dashboard.brain.persona == Persona.GOD_MODE

        dashboard.close()

    def test_08_master_dashboard_ai_move_trigger_with_windmouse(self, qapp):
        """Tests triggering move execution with automated WindMouse dispatching."""
        mock_mouse = HumanMouseController(backend=MockMouseController(fast_mode=True), fast_mode=True)
        dashboard = MasterDashboardWindow(mouse_controller=mock_mouse, headless=True)

        # Enable autobot
        dashboard.autobot_checkbox.setChecked(True)

        signal_received = []
        dashboard.move_executed_signal.connect(lambda d: signal_received.append(d))

        # Trigger AI move
        decision = dashboard.trigger_ai_move()
        assert decision["move"] is not None
        assert len(signal_received) == 1

        # Check telemetry was recorded
        snap = dashboard.get_state_snapshot()
        assert snap["last_telemetry"]["success"] is True
        assert snap["last_telemetry"]["total_trajectory_points"] > 10

        dashboard.close()
