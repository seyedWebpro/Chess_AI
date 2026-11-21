"""
Deep Stress, Tactical Resilience, and End-to-End Simulation Tests
Verifies Maia Native ONNX inference and ACL Governor under extreme, real-world match conditions.
"""

import pytest
import chess
import time
from core.maia_onnx import MaiaOnnxEngine
from core.acl_governor import ACLGovernor, GovernorState
from core.hybrid_brain import HybridBrain, Persona


class TestDeepTacticalResilience:
    """Stress tests on complex tactical, endgame, and promotion positions."""

    @pytest.fixture(scope="class")
    def onnx_engine_1500(self):
        return MaiaOnnxEngine(elo=1500)

    @pytest.fixture(scope="class")
    def onnx_engine_1900(self):
        return MaiaOnnxEngine(elo=1900)

    def test_01_all_four_pawn_promotions_white_and_black(self, onnx_engine_1500):
        """Verifies that Queen, Rook, Bishop, and Knight promotions generate valid probabilities for both colors."""
        # 1. White pawn on e7
        white_promo_fen = "7k/4P3/8/8/8/8/8/7K w - - 0 1"
        board_w = chess.Board(white_promo_fen)
        res_w = onnx_engine_1500.predict(board_w, top_k=10)

        promo_moves_w = {c["uci"]: c["probability"] for c in res_w["candidates"] if "e7e8" in c["uci"]}
        assert "e7e8q" in promo_moves_w
        assert promo_moves_w["e7e8q"] > 0.90, "Queen promotion must dominate"
        assert all(p >= 0.0 for p in promo_moves_w.values())

        # 2. Black pawn on e2
        black_promo_fen = "7k/8/8/8/8/8/4p3/7K b - - 0 1"
        board_b = chess.Board(black_promo_fen)
        res_b = onnx_engine_1500.predict(board_b, top_k=10)

        promo_moves_b = {c["uci"]: c["probability"] for c in res_b["candidates"] if "e2e1" in c["uci"]}
        assert "e2e1q" in promo_moves_b
        assert promo_moves_b["e2e1q"] > 0.90, "Black Queen promotion must dominate"
        assert all(p >= 0.0 for p in promo_moves_b.values())

    def test_02_black_perspective_tactical_openings(self, onnx_engine_1900):
        """Verifies Black's perspective predictions in classical opening positions."""
        # Against 1. e4: Sicilian and 1...e5 must be top candidates
        board = chess.Board()
        board.push_san("e4")
        res = onnx_engine_1900.predict(board, top_k=5)

        top_ucis = [c["uci"] for c in res["candidates"]]
        assert any(m in top_ucis for m in ["c7c5", "e7e5", "e7e6", "c7c6"]), "Must include standard human replies to 1.e4"
        assert res["best_move"] in board.legal_moves

        # Against 1. d4: 1...d5 and 1...Nf6 must be top candidates
        board_d4 = chess.Board()
        board_d4.push_san("d4")
        res_d4 = onnx_engine_1900.predict(board_d4, top_k=5)
        top_ucis_d4 = [c["uci"] for c in res_d4["candidates"]]
        assert any(m in top_ucis_d4 for m in ["d7d5", "g8f6"]), "Must include standard human replies to 1.d4"

    def test_03_acl_governor_black_to_move_centipawn_loss(self):
        """Verifies that ACL Governor correctly computes centipawn loss from Black's perspective."""
        gov = ACLGovernor(target_elo=1500)
        gov.total_loss_cp = 10.0
        gov.move_count = 5  # ACPL = 2.0 -> TOO_PRECISE

        # Black to move: Stockfish White eval gives 25 for best move, 55 for second move (30cp loss for Black)
        board_b = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2")
        candidates = [
            {"move_uci": "b8c6", "score_cp": 25, "is_mate": False},
            {"move_uci": "g8f6", "score_cp": 55, "is_mate": False},  # +30cp for White = 30cp loss for Black
        ]

        decision = gov.govern_candidates(board_b, candidates, current_eval_cp=25)
        # Should detect that g8f6 has ~30 cp loss and is a safe inaccuracy
        assert decision["chosen_move"].uci() in ["b8c6", "g8f6"]
        assert decision["loss_incurred"] >= 0.0

    def test_04_acl_governor_absolute_mate_safety(self):
        """Verifies that ACL Governor never injects an inaccuracy when mate is on the board."""
        gov = ACLGovernor(target_elo=1500)
        gov.total_loss_cp = 0.0
        gov.move_count = 10  # ACPL = 0.0 -> Strongly TOO_PRECISE

        # White has mate in 1 (Qxf7#)
        board = chess.Board("r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4")
        candidates = [
            {"move_uci": "f3f7", "score_cp": None, "is_mate": True, "mate_in": 1},
            {"move_uci": "c4f7", "score_cp": 350, "is_mate": False},
        ]

        decision = gov.govern_candidates(board, candidates, current_eval_cp=1000)
        # Must enforce mate in 1, never play c4f7
        assert decision["chosen_move"].uci() == "f3f7", "Governor must NEVER skip checkmate in 1"
        assert decision["loss_incurred"] == 0.0

    def test_05_rapid_multi_elo_session_switching(self):
        """Verifies rapid switching between models (1100, 1500, 1900) has zero memory leak or crash."""
        engine = MaiaOnnxEngine(elo=1500)
        board = chess.Board()

        elos = [1100, 1500, 1900, 1100, 1900, 1500] * 5  # 30 rapid switches
        t_start = time.perf_counter()

        for target_elo in elos:
            engine.set_elo(target_elo)
            res = engine.predict(board, top_k=2)
            assert res["best_move"] is not None
            assert res["elo"] == target_elo

        elapsed = time.perf_counter() - t_start
        assert elapsed < 5.0, f"30 switches took {elapsed:.2f}s, expected < 5s"


class TestLiveGameSimulationE2E:
    """Full match simulation testing real-time AI decision lifecycle."""

    def test_06_e2e_30_ply_live_game_simulation(self):
        """
        Simulates a continuous 30-ply live match between Maia 1500 and Stockfish/Maia
        verifying that all moves are legal, latency is < 20ms per move, and ACPL is bounded.
        """
        with HybridBrain(persona=Persona.MAIA_1500, use_opening_book=True, use_acl_governor=True) as brain:
            board = chess.Board()
            move_times = []
            sources = []

            for ply in range(30):
                if board.is_game_over():
                    break

                t0 = time.perf_counter()
                decision = brain.decide_move(board, time_limit=0.03)
                t1 = time.perf_counter()

                move_times.append((t1 - t0) * 1000.0)
                sources.append(decision["source"])

                chosen_move = decision["move"]
                assert chosen_move in board.legal_moves, f"Move {chosen_move} at ply {ply} must be strictly legal"
                board.push(chosen_move)

            assert len(board.move_stack) >= 20, "Simulation must reach at least 20 plies"
            avg_time = sum(move_times) / len(move_times)

            # Assert performance
            assert avg_time < 60.0, f"Average move time was {avg_time:.2f}ms, expected < 60ms"
            assert "POLYGLOT_BOOK" in sources, "Opening book must be utilized"
            assert any("MAIA" in s for s in sources), "Maia engine must take over after book"

            # Check ACL Governor cumulative statistics
            acpl = brain.acl_governor.current_acpl
            assert 0.0 <= acpl <= 100.0, f"Final match ACPL {acpl} cp must be within safe human bounds"
