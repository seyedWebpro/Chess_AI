"""
maia_engine.py: Maia Human-Predictive Neural Policy Engine & Rating Persona Models.
Simulates human psychological chess behavior across Elo 1100, 1500, 1900, and God Mode
using Boltzmann Softmax policy distributions, cognitive heuristics, and ONNX runtime support.
"""

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from pathlib import Path
import math
import random
import numpy as np
import chess
from loguru import logger

from core.engine_manager import EngineManager
from core.maia_onnx import MaiaOnnxEngine


class Persona(Enum):
    """Play personas representing different human rating levels and engine strengths."""
    GOD_MODE = "God Mode (Stockfish 19 - Elo 3600)"
    MAIA_1900 = "Maia 1900 (Club Master - Elo 1900)"
    MAIA_1500 = "Maia 1500 (Intermediate - Elo 1500)"
    MAIA_1100 = "Maia 1100 (Casual / Novice - Elo 1100)"
    BLUNDER_SIM = "Human Blunder Simulator"


# Persona configuration profiles
PERSONA_PROFILES: Dict[Persona, Dict[str, Any]] = {
    Persona.GOD_MODE: {
        "target_elo": 3600,
        "temperature": 0.001,      # Sharp argmax selection
        "depth": 14,
        "skill_level": 20,
        "blunder_rate": 0.0,
        "top_pv_count": 1,
    },
    Persona.MAIA_1900: {
        "target_elo": 1900,
        "temperature": 0.18,       # Sharp on top 1-2 moves
        "depth": 8,
        "skill_level": 16,
        "blunder_rate": 0.005,
        "top_pv_count": 3,
    },
    Persona.MAIA_1500: {
        "target_elo": 1500,
        "temperature": 0.55,       # Solid club player, occasional inaccuracy
        "depth": 5,
        "skill_level": 11,
        "blunder_rate": 0.025,
        "top_pv_count": 4,
    },
    Persona.MAIA_1100: {
        "target_elo": 1100,
        "temperature": 1.35,       # Human amateur, intuitive developing moves
        "depth": 3,
        "skill_level": 5,
        "blunder_rate": 0.08,
        "top_pv_count": 5,
    },
    Persona.BLUNDER_SIM: {
        "target_elo": 1300,
        "temperature": 1.10,
        "depth": 4,
        "skill_level": 8,
        "blunder_rate": 0.15,
        "top_pv_count": 5,
    },
}


def _normalize_score_cp(candidate: Dict[str, Any]) -> float:
    """Converts centipawns or mate scores to a continuous numeric scale."""
    if candidate.get("is_mate", False):
        mate_in = candidate.get("mate_in", 1) or 1
        return (20000.0 - abs(mate_in) * 100.0) if mate_in > 0 else (-20000.0 + abs(mate_in) * 100.0)
    score = candidate.get("score_cp")
    return float(score) if score is not None else 0.0


class MaiaEngine:
    """
    Maia Chess Human Behavioral Predictor & Persona Selector.
    Translates engine Multi-PV analysis into human-like probability distributions
    mirroring real human decisions at different Elo rating tiers.
    """

    def __init__(
        self,
        engine_manager: Optional[EngineManager] = None,
        persona: Persona = Persona.MAIA_1500,
        onnx_model_path: Optional[Path] = None,
        use_native_onnx: bool = True,
    ):
        self.engine = engine_manager
        self.owns_engine = False
        if self.engine is None:
            self.engine = EngineManager()
            self.owns_engine = True

        self.persona = persona
        self.onnx_model_path = onnx_model_path
        self.use_native_onnx = use_native_onnx
        self.onnx_engine: Optional[MaiaOnnxEngine] = None

        if self.use_native_onnx:
            self._init_onnx_engine()

    def _init_onnx_engine(self) -> None:
        """Initializes native ONNX Maia engine if weights are available."""
        profile = PERSONA_PROFILES.get(self.persona)
        if profile and self.persona != Persona.GOD_MODE:
            try:
                self.onnx_engine = MaiaOnnxEngine(elo=profile["target_elo"])
            except Exception as e:
                logger.warning(f"Native Maia ONNX engine unavailable ({e}), using Boltzmann fallback.")
                self.onnx_engine = None

    def set_persona(self, persona: Persona) -> None:
        """Dynamically switches active playing persona."""
        self.persona = persona
        profile = PERSONA_PROFILES[persona]
        logger.info(f"Switched persona to: {persona.value} (Target Elo: {profile['target_elo']})")
        if self.use_native_onnx and persona != Persona.GOD_MODE:
            if self.onnx_engine:
                self.onnx_engine.set_elo(profile["target_elo"])
            else:
                self._init_onnx_engine()

    def predict_move(
        self,
        board: chess.Board,
        time_limit: float = 0.10,
    ) -> Dict[str, Any]:
        """
        Predicts human move according to active Persona profile.
        
        Args:
            board: Current chess.Board position.
            time_limit: Time allocated for multi-PV engine scan.
            
        Returns:
            Dictionary with chosen move, persona, probability distribution, and human rationale.
        """
        if board.is_game_over():
            return {"move": None, "uci": None, "reason": "game_over"}

        profile = PERSONA_PROFILES[self.persona]
        pv_count = profile["top_pv_count"]
        temp = profile["temperature"]

        # Native Maia ONNX Inference Path
        if self.persona != Persona.GOD_MODE and self.use_native_onnx and self.onnx_engine is not None:
            try:
                res = self.onnx_engine.predict(board, temperature=temp)
                chosen_move = res["best_move"]
                prob_dict = {c["uci"]: round(c["probability"], 4) for c in res["candidates"]}
                score_cp = 0
                if self.engine:
                    try:
                        self.engine.set_fen(board.fen())
                        eval_data = self.engine.evaluate_position(time_limit=min(0.015, time_limit))
                        score_cp = eval_data.get("score_cp", 0)
                    except Exception:
                        pass
                return {
                    "move": chosen_move,
                    "uci": chosen_move.uci() if chosen_move else None,
                    "persona": self.persona.name,
                    "score_cp": score_cp,
                    "is_blunder": False,
                    "probabilities": prob_dict,
                    "pv_rank": 1,
                    "source": f"MAIA_ONNX_{self.onnx_engine.elo}",
                    "wdl": res.get("wdl"),
                }
            except Exception as e:
                logger.warning(f"Maia ONNX prediction failed ({e}), falling back to Boltzmann engine.")

        # Ensure board FEN is set in engine
        self.engine.set_fen(board.fen())

        # Retrieve top candidate moves
        candidate_moves = self.engine.get_top_moves(n=pv_count, time_limit=time_limit)

        if not candidate_moves:
            # Fallback to random legal move if engine produced no candidate
            fallback_move = random.choice(list(board.legal_moves))
            return {
                "move": fallback_move,
                "uci": fallback_move.uci(),
                "persona": self.persona.name,
                "score_cp": 0,
                "is_blunder": False,
                "probabilities": {fallback_move.uci(): 1.0},
            }

        if self.persona == Persona.GOD_MODE:
            # God Mode: Strictly pick PV 1 (Engine Best)
            best_cand = candidate_moves[0]
            chosen_move = chess.Move.from_uci(best_cand["move_uci"])
            return {
                "move": chosen_move,
                "uci": chosen_move.uci(),
                "persona": self.persona.name,
                "score_cp": best_cand["score_cp"],
                "is_blunder": False,
                "probabilities": {best_cand["move_uci"]: 1.0},
                "pv_rank": 1,
            }

        # Human Policy Boltzmann Softmax Calculation:
        # Convert scores into continuous numeric scale
        norm_scores = [_normalize_score_cp(c) for c in candidate_moves]
        best_score = norm_scores[0]
        moves_uci = [c["move_uci"] for c in candidate_moves]

        # Calculate logits relative to best move
        logits = []
        for s in norm_scores:
            delta_cp = s - best_score
            # Clip delta to avoid extreme exponential overflows
            scaled_delta = max(-800.0, min(0.0, float(delta_cp)))
            logits.append(scaled_delta / (100.0 * temp))

        # Numerically stable Softmax
        exp_logits = np.exp(np.array(logits) - np.max(logits))
        probabilities = (exp_logits / np.sum(exp_logits)).tolist()

        # Human Blunder Injection Check
        is_blunder = False
        blunder_threshold = profile["blunder_rate"]
        if random.random() < blunder_threshold and len(candidate_moves) > 1:
            # Human slips: picks a sub-optimal move (lower PV)
            is_blunder = True
            sub_optimal_indices = list(range(1, len(candidate_moves)))
            chosen_idx = random.choice(sub_optimal_indices)
        else:
            # Sample from human Boltzmann probability distribution
            chosen_idx = np.random.choice(len(candidate_moves), p=probabilities)

        chosen_cand = candidate_moves[chosen_idx]
        chosen_move = chess.Move.from_uci(chosen_cand["move_uci"])

        prob_dict = {
            m: round(float(p), 4) for m, p in zip(moves_uci, probabilities)
        }

        return {
            "move": chosen_move,
            "uci": chosen_move.uci(),
            "persona": self.persona.name,
            "score_cp": chosen_cand["score_cp"],
            "is_blunder": is_blunder,
            "probabilities": prob_dict,
            "pv_rank": chosen_idx + 1,
            "best_engine_move": candidate_moves[0]["move_uci"],
        }

    def close(self) -> None:
        if self.owns_engine and self.engine:
            self.engine.close()
            self.engine = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
