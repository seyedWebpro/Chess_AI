"""
maia_onnx.py: Native ONNX Runtime Inference Engine for Maia Chess.
Direct neural policy prediction for Elo 1100, 1500, and 1900 using the official
112-plane board encoding and 1858-dimensional policy vocabulary.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
import chess
import onnxruntime as ort
from loguru import logger

from core.policy_index import policy_index

# Pre-map UCI moves to policy indices
MOVE_TO_INDEX: Dict[str, int] = {m: i for i, m in enumerate(policy_index)}
INDEX_TO_MOVE: List[str] = policy_index

# Piece order for Lc0 112-plane encoding:
# Current player pieces: P, N, B, R, Q, K
# Opponent pieces: p, n, b, r, q, k
PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def encode_board_112_planes(board: chess.Board) -> np.ndarray:
    """
    Encodes a python-chess Board into the canonical Leela Chess Zero (Lc0)
    112-plane tensor of shape (1, 112, 8, 8) with dtype float32,
    oriented from the active player's perspective, including position history.
    """
    planes = np.zeros((1, 112, 8, 8), dtype=np.float32)
    turn = board.turn

    # 1. Historical Piece Planes (8 time steps x 13 planes = 104 planes)
    curr_board = board.copy()
    for step in range(8):
        base = step * 13
        for sq in chess.SQUARES:
            piece = curr_board.piece_at(sq)
            if piece is None:
                continue
            r = chess.square_rank(sq) if turn == chess.WHITE else 7 - chess.square_rank(sq)
            c = chess.square_file(sq)
            plane_idx = base + PIECE_ORDER.index(piece.piece_type)
            if piece.color != turn:
                plane_idx += 6
            planes[0, plane_idx, r, c] = 1.0

        if not curr_board.move_stack:
            break
        curr_board.pop()

    # 2. Auxiliary Planes (Planes 104 to 111)
    # Castling Rights:
    # 104: our queenside, 105: our kingside, 106: opp queenside, 107: opp kingside
    if turn == chess.WHITE:
        if board.has_queenside_castling_rights(chess.WHITE): planes[0, 104, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.WHITE): planes[0, 105, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK): planes[0, 106, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK): planes[0, 107, :, :] = 1.0
    else:
        if board.has_queenside_castling_rights(chess.BLACK): planes[0, 104, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK): planes[0, 105, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE): planes[0, 106, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.WHITE): planes[0, 107, :, :] = 1.0

    # 108: En-passant square
    if board.ep_square is not None:
        ep_r = chess.square_rank(board.ep_square) if turn == chess.WHITE else 7 - chess.square_rank(board.ep_square)
        ep_c = chess.square_file(board.ep_square)
        planes[0, 108, ep_r, ep_c] = 1.0

    # 109: Rule-50 counter
    planes[0, 109, :, :] = float(board.halfmove_clock) / 100.0

    # 111: Constant Plane of 1s
    planes[0, 111, :, :] = 1.0

    return planes


class MaiaOnnxEngine:
    """
    Executes native ONNX deep learning inference for Maia Chess models.
    Supports Maia 1100, 1500, and 1900.
    """

    DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "data" / "models" / "maia"

    def __init__(
        self,
        elo: int = 1500,
        model_dir: Optional[Path] = None,
    ):
        self.elo = elo
        self.model_dir = model_dir or self.DEFAULT_MODEL_DIR
        self.sessions: Dict[int, ort.InferenceSession] = {}
        self._load_session(elo)

    def _get_model_path(self, elo: int) -> Path:
        closest = 1500
        if elo <= 1300:
            closest = 1100
        elif elo >= 1700:
            closest = 1900
        return self.model_dir / f"maia-{closest}.onnx"

    def _load_session(self, elo: int) -> Optional[ort.InferenceSession]:
        path = self._get_model_path(elo)
        if not path.exists():
            logger.warning(f"Maia ONNX weight file not found at: {path}")
            return None

        if elo not in self.sessions:
            try:
                # Use CPU provider with fast threading
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                opts.inter_op_num_threads = 1
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
                self.sessions[elo] = session
                logger.success(f"Loaded native Maia ONNX model ({path.name})")
            except Exception as e:
                logger.error(f"Failed to load ONNX model {path}: {e}")
                return None

        return self.sessions[elo]

    def set_elo(self, elo: int) -> None:
        """Switches active rating level."""
        self.elo = elo
        self._load_session(elo)

    def predict(
        self,
        board: chess.Board,
        temperature: float = 0.5,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Runs native ONNX neural inference on the board.
        Returns top legal moves with predicted human probabilities and WDL win rate.
        """
        if board.is_game_over():
            return {"best_move": None, "moves": [], "wdl": [0.33, 0.34, 0.33]}

        session = self._load_session(self.elo)
        if session is None:
            raise FileNotFoundError(f"Maia ONNX model not loaded for Elo {self.elo}")

        # 1. Encode 112 input planes
        input_tensor = encode_board_112_planes(board)

        # 2. Run ONNX Inference
        outputs = session.run(None, {"/input/planes": input_tensor})
        policy_logits = outputs[0][0]
        wdl = outputs[1][0]  # [Win, Draw, Loss] for current player

        # 3. Extract and filter to legal moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {"best_move": None, "moves": [], "wdl": wdl.tolist()}

        move_candidates = []
        is_white = board.turn == chess.WHITE

        for m in legal_moves:
            from_sq = m.from_square if is_white else chess.square_mirror(m.from_square)
            to_sq = m.to_square if is_white else chess.square_mirror(m.to_square)
            promo = m.promotion

            uci_oriented = chess.square_name(from_sq) + chess.square_name(to_sq)
            if promo:
                uci_oriented += chess.piece_symbol(promo).lower()

            if uci_oriented in MOVE_TO_INDEX:
                idx = MOVE_TO_INDEX[uci_oriented]
                logit = float(policy_logits[idx])
            elif promo == chess.KNIGHT:
                # Underpromotion to Knight fallback: use queen promotion logit minus underpromotion penalty
                queen_uci = uci_oriented[:4] + "q"
                base_uci = uci_oriented[:4]
                if queen_uci in MOVE_TO_INDEX:
                    logit = float(policy_logits[MOVE_TO_INDEX[queen_uci]]) - 4.5
                elif base_uci in MOVE_TO_INDEX:
                    logit = float(policy_logits[MOVE_TO_INDEX[base_uci]]) - 4.5
                else:
                    logit = -100.0
            else:
                logit = -100.0

            move_candidates.append({
                "move": m,
                "uci": m.uci(),
                "logit": logit,
            })

        # 4. Softmax over legal moves with temperature scaling
        logits = np.array([c["logit"] for c in move_candidates], dtype=np.float64)
        scaled_logits = logits / max(0.01, temperature)
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exp_logits / np.sum(exp_logits)

        for i, c in enumerate(move_candidates):
            c["probability"] = float(probs[i])

        # Sort descending by probability
        move_candidates.sort(key=lambda x: x["probability"], reverse=True)

        best_cand = move_candidates[0]

        return {
            "best_move": best_cand["move"],
            "best_move_uci": best_cand["uci"],
            "probability": best_cand["probability"],
            "candidates": move_candidates[:top_k],
            "wdl": {
                "win": float(wdl[0]),
                "draw": float(wdl[1]),
                "loss": float(wdl[2]),
            },
            "elo": self.elo,
            "source": f"MAIA_ONNX_{self.elo}",
        }
