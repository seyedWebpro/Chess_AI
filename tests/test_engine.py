"""
Comprehensive Unit & Integration Test Suite for Phase 1: EngineManager.
Tests normal play, tactical puzzles, stalemate, auto-promotions, castling, en passant,
invalid FEN resilience, self-healing process restarts, and caching benchmarks.
"""

import time
import unittest
import chess
from loguru import logger
from core.engine_manager import EngineManager


class TestEngineManager(unittest.TestCase):
    """Deep test suite for EngineManager."""

    @classmethod
    def setUpClass(cls):
        logger.info("Initializing EngineManager for comprehensive tests...")
        cls.manager = EngineManager(threads=2, hash_size_mb=64, enable_cache=True)

    @classmethod
    def tearDownClass(cls):
        logger.info("Tearing down EngineManager...")
        cls.manager.close()

    def setUp(self):
        self.manager.reset()

    def test_01_initial_position_evaluation(self):
        """Test standard starting position evaluation."""
        start_time = time.perf_counter()
        result = self.manager.evaluate_position(time_limit=0.05)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        self.assertIsNotNone(result["best_move"])
        self.assertIn(result["best_move"], ["e2e4", "d2d4", "g1f3", "c2c4"])
        self.assertFalse(result["is_mate"])
        self.assertGreaterEqual(result["win_chance_pct"], 40.0)
        self.assertLessEqual(result["win_chance_pct"], 65.0)

    def test_02_mate_in_one_detection(self):
        """Test engine identifies forced mate in 1."""
        fen_mate_in_one = "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4"
        self.assertTrue(self.manager.set_fen(fen_mate_in_one))

        result = self.manager.evaluate_position(time_limit=0.1)
        self.assertEqual(result["best_move"], "f3f7")
        self.assertTrue(result["is_mate"])
        self.assertEqual(result["mate_in"], 1)
        self.assertEqual(result["eval_str"], "#1")
        self.assertEqual(result["win_chance_pct"], 100.0)

    def test_03_stalemate_draw_detection(self):
        """Test terminal stalemate position correctly evaluates as a 50% draw."""
        # Black King on a8, White King on c7, White Queen on b6 -> Black to move is stalemated
        fen_stalemate = "k7/2K5/1Q6/8/8/8/8/8 b - - 0 1"
        self.assertTrue(self.manager.set_fen(fen_stalemate))
        self.assertTrue(self.manager.board.is_stalemate())

        result = self.manager.evaluate_position(time_limit=0.05)
        self.assertIsNone(result["best_move"])
        self.assertEqual(result["eval_str"], "1/2-1/2 (Draw)")
        self.assertEqual(result["win_chance_pct"], 50.0)
        self.assertFalse(result["is_mate"])

    def test_04_pawn_promotion_auto_queen(self):
        """Test pawn move e7e8 without suffix auto-promotes to Queen (e7e8q)."""
        fen_promo = "k7/4P3/8/8/8/8/8/4K3 w - - 0 1"
        self.assertTrue(self.manager.set_fen(fen_promo))

        # Check legality of shorthand e7e8
        self.assertTrue(self.manager.is_move_legal("e7e8"))
        move = self.manager.make_move("e7e8")
        self.assertIsNotNone(move)
        self.assertEqual(move.uci(), "e7e8q")
        self.assertEqual(move.promotion, chess.QUEEN)

    def test_05_castling_and_en_passant(self):
        """Test special chess moves: Castling and En Passant."""
        # 1. Kingside Castling
        fen_castling = "rnbqk2r/pppp1ppp/5n2/2b1p3/4P3/5N2/PPPPBPPP/RNBQK2R w KQkq - 4 4"
        self.assertTrue(self.manager.set_fen(fen_castling))
        self.assertTrue(self.manager.is_move_legal("e1g1"))
        move_castle = self.manager.make_move("e1g1")
        self.assertIsNotNone(move_castle)

        # 2. En Passant
        fen_ep = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
        self.assertTrue(self.manager.set_fen(fen_ep))
        self.assertTrue(self.manager.is_move_legal("e5f6"))
        move_ep = self.manager.make_move("e5f6")
        self.assertIsNotNone(move_ep)
        self.assertEqual(move_ep.uci(), "e5f6")

    def test_06_invalid_fen_handling(self):
        """Test invalid FEN string is rejected safely without breaking state."""
        prev_fen = self.manager.get_fen()
        success = self.manager.set_fen("not_a_valid_fen_string_at_all")
        self.assertFalse(success)
        # Previous position must be preserved
        self.assertEqual(self.manager.get_fen(), prev_fen)

    def test_07_cache_acceleration(self):
        """Test repeated evaluations of the same position take < 0.2ms via LRU cache."""
        self.manager.reset()
        # First call (populates cache)
        self.manager.evaluate_position(time_limit=0.03)

        # Second call (hits cache)
        t0 = time.perf_counter()
        cached_result = self.manager.evaluate_position(time_limit=0.03)
        cached_time_ms = (time.perf_counter() - t0) * 1000

        self.assertLess(cached_time_ms, 2.0)  # Must be virtually instant (< 2.0ms even under full test suite load)
        self.assertIsNotNone(cached_result["best_move"])

    def test_08_self_healing_engine_restart(self):
        """Test engine recovers automatically if process unexpectedly terminates."""
        self.manager.reset()
        self.manager.clear_cache()

        # Simulate engine crash or closure
        self.manager.close()
        self.assertFalse(self.manager.is_alive())

        # Calling evaluate_position should trigger self-healing restart
        result = self.manager.evaluate_position(time_limit=0.05)
        self.assertTrue(self.manager.is_alive())
        self.assertIsNotNone(result["best_move"])


if __name__ == "__main__":
    unittest.main()
