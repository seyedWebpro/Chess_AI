"""
test_maia_onnx.py: Unit and Integration Test Suite for Native Maia ONNX Engine.
Verifies 112-plane board encoding, ONNX neural inference, WDL predictions, and Elo persona scaling.
"""

import pytest
import numpy as np
import chess
from core.maia_onnx import MaiaOnnxEngine, encode_board_112_planes


@pytest.fixture(scope="module")
def maia_1500():
    return MaiaOnnxEngine(elo=1500)


def test_01_encode_board_112_planes():
    board = chess.Board()
    tensor = encode_board_112_planes(board)

    assert tensor.shape == (1, 112, 8, 8)
    assert tensor.dtype == np.float32

    # Verify constant plane 111 is all 1.0
    assert np.all(tensor[0, 111, :, :] == 1.0)

    # Verify white pawns (plane 0) are on rank 1 (0-indexed)
    assert np.all(tensor[0, 0, 1, :] == 1.0)

    # Verify black pawns (plane 6) are on rank 6
    assert np.all(tensor[0, 6, 6, :] == 1.0)


def test_02_maia_1500_inference_starting_position(maia_1500):
    board = chess.Board()
    res = maia_1500.predict(board, temperature=0.5, top_k=5)

    assert res["best_move"] is not None
    assert res["best_move"].uci() in ["e2e4", "d2d4", "c2c4", "g1f3"]
    assert res["probability"] > 0.05
    assert len(res["candidates"]) == 5

    # WDL check
    wdl = res["wdl"]
    assert "win" in wdl and "draw" in wdl and "loss" in wdl
    total_wdl = wdl["win"] + wdl["draw"] + wdl["loss"]
    assert 0.98 <= total_wdl <= 1.02


def test_03_maia_black_perspective_response(maia_1500):
    # 1. e4
    board = chess.Board()
    board.push_san("e4")

    res = maia_1500.predict(board, temperature=0.5, top_k=5)
    candidate_ucis = [c["uci"] for c in res["candidates"]]

    # Common human responses to 1.e4 at Elo 1500: e7e5, c7c5, d7d5, e7e6
    assert any(m in candidate_ucis for m in ["e7e5", "c7c5", "d7d5", "e7e6"])


def test_04_maia_multi_elo_models():
    # Test loading Maia 1100 and Maia 1900
    engine_1100 = MaiaOnnxEngine(elo=1100)
    engine_1900 = MaiaOnnxEngine(elo=1900)

    board = chess.Board()
    res_1100 = engine_1100.predict(board)
    res_1900 = engine_1900.predict(board)

    assert res_1100["best_move"] in board.legal_moves
    assert res_1900["best_move"] in board.legal_moves
