"""
test_acl_governor.py: Unit and Integration Test Suite for Adaptive Centipawn Loss (ACL) Governor.
Verifies cumulative ACPL tracking, safe inaccuracy injection, error recovery, and tactical emergency safeguards.
"""

import pytest
import chess
from core.acl_governor import ACLGovernor, GovernorState, RATING_ACL_PROFILES


@pytest.fixture
def governor_1500():
    return ACLGovernor(target_elo=1500)


def test_01_acpl_tracking_and_reset(governor_1500):
    assert governor_1500.move_count == 0
    assert governor_1500.current_acpl == 40.0 # Initial default

    governor_1500.record_move_loss(30.0)
    assert governor_1500.move_count == 1
    assert governor_1500.current_acpl == 30.0

    governor_1500.record_move_loss(50.0)
    assert governor_1500.move_count == 2
    assert governor_1500.current_acpl == 40.0

    governor_1500.reset()
    assert governor_1500.move_count == 0
    assert len(governor_1500.loss_history) == 0


def test_02_governor_too_precise_triggers_safe_inaccuracy(governor_1500):
    # Simulate a game where player was too accurate (e.g. 5 moves with 0 loss)
    for _ in range(5):
        governor_1500.record_move_loss(2.0)

    # Current ACPL is 2.0 (well below min_acpl of 32.0)
    assert governor_1500.get_state() == GovernorState.TOO_PRECISE

    board = chess.Board()
    # Mock engine candidate moves:
    # PV 1: e2e4 (eval +50cp, loss 0)
    # PV 2: d2d4 (eval +25cp, loss 25cp -> safe inaccuracy)
    # PV 3: a2a3 (eval -150cp, loss 200cp -> blunder)
    candidates = [
        {"move_uci": "e2e4", "score_cp": 50, "is_mate": False},
        {"move_uci": "d2d4", "score_cp": 25, "is_mate": False},
        {"move_uci": "a2a3", "score_cp": -150, "is_mate": False},
    ]

    res = governor_1500.govern_candidates(board, candidates, current_eval_cp=50)

    # Should select d2d4 to increase ACPL safely without blundering!
    assert res["chosen_candidate"]["move_uci"] == "d2d4"
    assert "SAFE_INACCURACY" in res["governed_action"]
    assert res["loss_incurred"] == 25.0


def test_03_governor_too_inaccurate_enforces_best_move(governor_1500):
    # Simulate a game where player was inaccurate (5 moves with 70 loss)
    for _ in range(5):
        governor_1500.record_move_loss(70.0)

    assert governor_1500.get_state() == GovernorState.TOO_INACCURATE

    board = chess.Board()
    candidates = [
        {"move_uci": "e2e4", "score_cp": 50, "is_mate": False},
        {"move_uci": "d2d4", "score_cp": 20, "is_mate": False},
    ]

    res = governor_1500.govern_candidates(board, candidates, current_eval_cp=50)

    # Should strictly enforce PV 1 to recover accuracy
    assert res["chosen_candidate"]["move_uci"] == "e2e4"
    assert "ENFORCE_BEST_RECOVERY" in res["governed_action"]


def test_04_emergency_safeguards_override_inaccuracy(governor_1500):
    # Even if TOO_PRECISE, if under check or mate, governor MUST play best move
    for _ in range(5):
        governor_1500.record_move_loss(2.0)
    assert governor_1500.get_state() == GovernorState.TOO_PRECISE

    # Setup board under check
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert board.is_check()

    candidates = [
        {"move_uci": "g2g3", "score_cp": -200, "is_mate": False},
    ]

    res = governor_1500.govern_candidates(board, candidates, current_eval_cp=-200)
    # Never injects inaccuracies during tactical emergency
    assert res["chosen_candidate"]["move_uci"] == "g2g3"


def test_05_dynamic_elo_profile_switch(governor_1500):
    governor_1500.set_target_elo(1900)
    assert governor_1500.profile["target_acpl"] == 25.0

    governor_1500.set_target_elo(1100)
    assert governor_1500.profile["target_acpl"] == 65.0
