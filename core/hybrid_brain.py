"""
hybrid_brain.py: Hybrid AI Brain Core (Dual-Brain Engine & Opening Book Orchestrator).
Combines Polyglot Opening Book, Stockfish 19 NNUE (God Mode), and Maia Rating Personas
with cognitive deliberation time modeling.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import chess
from loguru import logger

from core.engine_manager import EngineManager
from core.opening_book import PolyglotBookManager
from core.maia_engine import MaiaEngine, Persona, PERSONA_PROFILES
from core.acl_governor import ACLGovernor, GovernorState
from automation.human_behavior import HumanThinkTimer


class HybridBrain:
    """
    Central Decision-Making Brain of the Chess AI system.
    Orchestrates Polyglot opening books (0ms lookup), Native Maia ONNX neural models,
    Stockfish 19 NNUE evaluation, Adaptive Centipawn Loss (ACL) Governance,
    and human cognitive reaction delays.
    """

    def __init__(
        self,
        persona: Persona = Persona.MAIA_1500,
        use_opening_book: bool = True,
        max_book_ply: int = 16,
        use_acl_governor: bool = True,
        engine_path: Optional[str] = None,
        book_path: Optional[Path] = None,
    ):
        self.persona = persona
        self.use_opening_book = use_opening_book
        self.max_book_ply = max_book_ply
        self.use_acl_governor = use_acl_governor

        self.engine = EngineManager(engine_path=engine_path)
        self.book_manager = PolyglotBookManager(book_path=book_path)
        self.maia = MaiaEngine(engine_manager=self.engine, persona=persona, use_native_onnx=True)
        self.timer = HumanThinkTimer()

        target_elo = PERSONA_PROFILES.get(persona, {}).get("target_elo", 1500)
        self.acl_governor = ACLGovernor(target_elo=target_elo)

    def set_persona(self, persona: Persona) -> None:
        """Updates active playing persona and reconfigures ACL governor."""
        self.persona = persona
        self.maia.set_persona(persona)
        target_elo = PERSONA_PROFILES.get(persona, {}).get("target_elo", 1500)
        self.acl_governor.set_target_elo(target_elo)

    def reset_game(self) -> None:
        """Resets game statistics in ACL governor."""
        self.acl_governor.reset()

    def set_opening_book_enabled(self, enabled: bool) -> None:
        """Enables or disables the Polyglot opening book."""
        self.use_opening_book = enabled
        logger.info(f"Opening book enabled: {enabled}")

    def decide_move(
        self,
        board: chess.Board,
        time_limit: float = 0.10,
    ) -> Dict[str, Any]:
        """
        Determines the optimal move based on game state, opening book, and persona.
        
        Args:
            board: Current chess.Board.
            time_limit: Maximum engine search time in seconds.
            
        Returns:
            Dictionary containing chosen move, evaluation, source, and cognitive delay.
        """
        if board.is_game_over() or not board.is_valid():
            return {
                "move": None,
                "uci": None,
                "source": "GAME_OVER" if board.is_game_over() else "INVALID_BOARD",
                "is_book": False,
                "eval_str": "Game Over" if board.is_game_over() else "Invalid",
                "deliberation_time_sec": 0.0,
            }

        # 1. Polyglot Opening Book Phase
        if self.use_opening_book and board.ply() < self.max_book_ply:
            book_move = self.book_manager.get_book_move(board, mode="weighted")
            if book_move and book_move in board.legal_moves:
                delay = self.timer.calculate_delay(
                    is_opening_book=True,
                    ply_count=board.ply() + 1,
                )
                logger.debug(f"Book move selected: {book_move.uci()} (Ply {board.ply() + 1})")
                return {
                    "move": book_move,
                    "uci": book_move.uci(),
                    "source": "POLYGLOT_BOOK",
                    "persona": self.persona.name,
                    "eval_str": "Book Theory",
                    "is_book": True,
                    "deliberation_time_sec": delay,
                    "score_cp": 0,
                    "win_chance_pct": 50.0,
                    "is_blunder": False,
                }

        # 2. Maia & Stockfish Persona Inference Phase
        prediction = self.maia.predict_move(board, time_limit=time_limit)
        chosen_move = prediction["move"]

        governor_info = {}
        # Apply Adaptive ACL Governance if enabled (in human personas)
        if self.use_acl_governor and self.persona != Persona.GOD_MODE:
            try:
                self.engine.set_fen(board.fen())
                candidates = self.engine.get_top_moves(n=4, time_limit=min(0.03, time_limit))
                if candidates:
                    score_cp = candidates[0].get("score_cp") if candidates[0].get("score_cp") is not None else prediction.get("score_cp")
                    gov_res = self.acl_governor.govern_candidates(board, candidates, current_eval_cp=score_cp)
                    if gov_res.get("chosen_move"):
                        chosen_move = gov_res["chosen_move"]
                        governor_info = {
                            "acl_state": gov_res["governor_state"],
                            "current_acpl": gov_res["current_acpl"],
                            "target_acpl": gov_res["target_acpl"],
                            "governed_action": gov_res["governed_action"],
                            "loss_incurred": gov_res["loss_incurred"],
                        }
            except Exception as e:
                logger.debug(f"ACL Governor evaluation: {e}")

        # Check tactical context for cognitive reaction time
        is_in_check = board.is_check()
        is_recapture = False
        if board.move_stack:
            last_move = board.peek()
            if board.is_capture(last_move) and chosen_move and chosen_move.to_square == last_move.to_square:
                is_recapture = True

        delay = self.timer.calculate_delay(
            is_opening_book=False,
            is_recapture=is_recapture,
            is_in_check=is_in_check,
            is_blunder_check=prediction.get("score_cp") is not None and abs(prediction.get("score_cp", 0)) > 300,
            ply_count=board.ply() + 1,
        )

        score_cp = prediction.get("score_cp")
        if score_cp is not None:
            eval_str = f"{'+' if score_cp > 0 else ''}{score_cp / 100:.2f}"
            win_chance = round(100 / (1 + 10 ** (-score_cp / 400)), 1)
        else:
            eval_str = "+M1" if prediction.get("pv_rank") == 1 else "0.00"
            win_chance = 100.0 if prediction.get("pv_rank") == 1 else 50.0

        decision_dict = {
            "move": chosen_move,
            "uci": chosen_move.uci() if chosen_move else None,
            "source": prediction.get("source", "MAIA_ONNX" if self.persona != Persona.GOD_MODE else "STOCKFISH_GOD_MODE"),
            "persona": self.persona.name,
            "eval_str": eval_str,
            "score_cp": score_cp,
            "win_chance_pct": win_chance,
            "is_book": False,
            "is_blunder": prediction.get("is_blunder", False),
            "deliberation_time_sec": delay,
            "pv_rank": prediction.get("pv_rank", 1),
            "probabilities": prediction.get("probabilities", {}),
        }
        decision_dict.update(governor_info)
        return decision_dict

    def close(self) -> None:
        if self.maia:
            self.maia.close()
        if self.book_manager:
            self.book_manager.close()
        if self.engine:
            self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
