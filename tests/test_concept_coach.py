"""
test_concept_coach.py: Comprehensive test suite for AI Grandmaster Concept Coach (Phase 11).
Tests pin detection, knight forks, castling, promotions, checks, center outposts, and Persian text generation.
"""

import pytest
import chess
from core.concept_coach import ConceptCoach, TacticalTheme


@pytest.fixture
def coach():
    return ConceptCoach()


def test_opening_development(coach):
    board = chess.Board()
    # 1. e4
    e4 = chess.Move.from_uci("e2e4")
    res_e4 = coach.analyze_move(board, e4)
    assert TacticalTheme.CENTER_CONTROL in res_e4["themes"] or TacticalTheme.PAWN_BREAK in res_e4["themes"]
    assert "مرکز" in res_e4["short_concept"] or "پیاده" in res_e4["short_concept"]

    board.push(e4)
    # 1... e5
    board.push(chess.Move.from_uci("e7e5"))
    # 2. Nf3 (Development towards center)
    nf3 = chess.Move.from_uci("g1f3")
    res_nf3 = coach.analyze_move(board, nf3)
    assert TacticalTheme.DEVELOPMENT in res_nf3["themes"]
    assert "گسترش" in res_nf3["short_concept"]
    assert "اسب" in res_nf3["explanation_fa"]


def test_pin_detection(coach):
    # Setup pin position: 1. e4 e5 2. Nf3 d6 3. Bc4 Nc6 4. d4 ... or bishop pins knight on c6 after d6
    board = chess.Board("r1bqkbnr/ppp2ppp/2np4/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4")
    bb5 = chess.Move.from_uci("f1b5")
    res = coach.analyze_move(board, bb5)
    # Bb5 pins Nc6 to King on e8
    assert TacticalTheme.PIN in res["themes"]
    assert "آچمز" in res["short_concept"]
    assert "اسب" in res["explanation_fa"]


def test_relative_pin_to_queen(coach):
    # Setup pin: Black has pawn on e6, White plays Bg5 pinning Black's Nf6 to Queen on d8
    board = chess.Board("r1bqkb1r/pppp1ppp/2n1pn2/8/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4")
    bg5 = chess.Move.from_uci("c1g5")
    res = coach.analyze_move(board, bg5)
    assert TacticalTheme.PIN in res["themes"]
    assert "آچمز" in res["short_concept"]
    assert "اسب" in res["explanation_fa"]


def test_royal_fork_detection(coach):
    # Setup Knight fork attacking King and Queen: e.g. White Knight on e7 checks King on g8 and Queen on c8
    # FEN with King on g8, Queen on c8, White Knight jumping to e7
    board = chess.Board("2q3k1/4bppp/8/8/8/5N2/5PPP/6K1 w - - 0 1")
    # Move knight to e7+ -> forks King on g8 and Queen on c8!
    # Wait, e7 is attacked by nothing, lets place knight on d5 jumping to e7+
    board.set_piece_at(chess.D5, chess.Piece(chess.KNIGHT, chess.WHITE))
    ne7 = chess.Move.from_uci("d5e7")
    res = coach.analyze_move(board, ne7)
    assert TacticalTheme.FORK in res["themes"]
    assert TacticalTheme.CHECK in res["themes"]
    assert "چنگال" in res["short_concept"]


def test_castling_safety(coach):
    board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5")
    # O-O (White short castling)
    castle_move = chess.Move.from_uci("e1g1")
    res = coach.analyze_move(board, castle_move)
    assert TacticalTheme.CASTLING in res["themes"]
    assert "قلعه کوچک" in res["short_concept"]
    assert "امنیت" in res["explanation_fa"]


def test_pawn_promotion(coach):
    # Pawn on a7 about to promote to Queen
    board = chess.Board("8/P7/8/8/8/8/k7/7K w - - 0 1")
    promote_move = chess.Move.from_uci("a7a8q")
    res = coach.analyze_move(board, promote_move)
    assert TacticalTheme.PROMOTION in res["themes"]
    assert "ارتقای پیاده" in res["short_concept"]
    assert "وزیر" in res["explanation_fa"]


def test_checkmate_move(coach):
    # Scholar's mate: Qxf7#
    board = chess.Board("r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4")
    mate_move = chess.Move.from_uci("f3f7")
    res = coach.analyze_move(board, mate_move)
    assert TacticalTheme.CHECKMATE in res["themes"]
    assert "مات قطعی" in res["short_concept"]
