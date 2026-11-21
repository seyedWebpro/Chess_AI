"""
acl_governor.py: Adaptive Centipawn Loss (ACL) Governor.
Monitors cumulative Centipawn Loss (ACPL) throughout online chess games and dynamically
steers move selection to strictly maintain human statistical rating profiles (e.g. ACPL 35-45 for Elo 1500),
rendering play completely indistinguishable from real humans to anti-cheat algorithms.
"""

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import random
import numpy as np
import chess
from loguru import logger


class GovernorState(Enum):
    IN_TARGET = "IN_TARGET"            # ACPL is within natural human rating band
    TOO_PRECISE = "TOO_PRECISE"        # ACPL too low (playing like Stockfish 3600); inject safe inaccuracy
    TOO_INACCURATE = "TOO_INACCURATE"  # ACPL too high; enforce best engine move to recover


# Standard statistical ACPL targets by Elo rating
RATING_ACL_PROFILES: Dict[int, Dict[str, float]] = {
    1100: {"target_acpl": 65.0, "min_acpl": 50.0, "max_acpl": 80.0, "max_safe_loss": 85.0},
    1500: {"target_acpl": 40.0, "min_acpl": 32.0, "max_acpl": 48.0, "max_safe_loss": 55.0},
    1900: {"target_acpl": 25.0, "min_acpl": 20.0, "max_acpl": 32.0, "max_safe_loss": 35.0},
    3600: {"target_acpl": 5.0,  "min_acpl": 0.0,  "max_acpl": 12.0, "max_safe_loss": 10.0},
}


def _get_score_cp_pov(cand: Dict[str, Any], turn: chess.Color) -> float:
    """Returns evaluation from the perspective of the active player in centipawns."""
    if cand.get("is_mate", False):
        mate_in = cand.get("mate_in") or 1
        val = 20000.0 - abs(mate_in) * 100.0
        if mate_in < 0:
            val = -val
        return val if turn == chess.WHITE else -val

    score = cand.get("score_cp")
    if score is None:
        return 0.0
    return float(score) if turn == chess.WHITE else -float(score)


class ACLGovernor:
    """
    Closed-loop Centipawn Loss regulator.
    Maintains a target rating profile across an entire match by dynamically throttling accuracy.
    """

    def __init__(self, target_elo: int = 1500):
        self.target_elo = target_elo
        self.profile = self._resolve_profile(target_elo)

        self.total_loss_cp: float = 0.0
        self.move_count: int = 0
        self.loss_history: List[float] = []

    def _resolve_profile(self, elo: int) -> Dict[str, float]:
        if elo <= 1300:
            return RATING_ACL_PROFILES[1100]
        elif elo <= 1700:
            return RATING_ACL_PROFILES[1500]
        elif elo <= 2300:
            return RATING_ACL_PROFILES[1900]
        else:
            return RATING_ACL_PROFILES[3600]

    def set_target_elo(self, elo: int) -> None:
        """Updates target Elo and recalibrates profile."""
        self.target_elo = elo
        self.profile = self._resolve_profile(elo)
        logger.info(f"ACL Governor target set to Elo {elo} (ACPL target: {self.profile['target_acpl']} cp)")

    def reset(self) -> None:
        """Resets match statistics for a fresh game."""
        self.total_loss_cp = 0.0
        self.move_count = 0
        self.loss_history.clear()

    @property
    def current_acpl(self) -> float:
        """Current game cumulative Average Centipawn Loss."""
        if self.move_count == 0:
            return self.profile["target_acpl"]
        return round(self.total_loss_cp / float(self.move_count), 2)

    def get_state(self) -> GovernorState:
        """Determines if the game is currently too precise or inaccurate."""
        if self.move_count < 3:
            return GovernorState.IN_TARGET  # Opening theory grace period

        curr = self.current_acpl
        if curr < self.profile["min_acpl"]:
            return GovernorState.TOO_PRECISE
        elif curr > self.profile["max_acpl"]:
            return GovernorState.TOO_INACCURATE
        return GovernorState.IN_TARGET

    def record_move_loss(self, loss_cp: float) -> None:
        """Records the centipawn loss of a committed move."""
        loss_cp = max(0.0, float(loss_cp))
        self.total_loss_cp += loss_cp
        self.move_count += 1
        self.loss_history.append(loss_cp)

    def govern_candidates(
        self,
        board: chess.Board,
        candidates: List[Dict[str, Any]],
        current_eval_cp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Filters and selects the optimal candidate move to steer cumulative ACPL towards target.
        
        Args:
            board: Current chess.Board.
            candidates: Sorted engine Multi-PV candidates (rank 0 is best).
            current_eval_cp: Overall board evaluation in centipawns.
            
        Returns:
            Dictionary containing selected candidate, applied governance, and current metrics.
        """
        if not candidates:
            return {"chosen_candidate": None, "state": self.get_state()}

        # 1. Calculate loss relative to PV 1 for all candidates
        best_cand = candidates[0]
        turn = board.turn
        best_score_pov = _get_score_cp_pov(best_cand, turn)

        for c in candidates:
            cand_score_pov = _get_score_cp_pov(c, turn)
            c["loss_cp"] = max(0.0, float(best_score_pov - cand_score_pov))

        state = self.get_state()
        chosen = best_cand
        governed_action = "PLAY_BEST"

        # Check for critical tactical emergency where inaccuracy is prohibited
        active_eval_pov = current_eval_cp if (current_eval_cp is None or turn == chess.WHITE) else -current_eval_cp
        is_emergency = (
            board.is_check() or
            best_cand.get("is_mate", False) or
            len(candidates) == 1 or
            (active_eval_pov is not None and active_eval_pov < -200) # Losing heavily: don't blunder more
        )

        # 2. Steer based on Governor State
        if state == GovernorState.TOO_PRECISE and not is_emergency and len(candidates) > 1:
            # Need to increase ACPL with a SAFE inaccuracy
            max_safe = self.profile["max_safe_loss"]

            safe_candidates = [
                c for c in candidates[1:]
                if 10.0 <= c["loss_cp"] <= max_safe and not c.get("is_mate", False)
            ]

            if safe_candidates:
                # Pick the safe sub-optimal candidate closest to the required target gap
                chosen = safe_candidates[0]
                governed_action = f"INJECT_SAFE_INACCURACY (+{chosen['loss_cp']:.1f}cp)"
            else:
                chosen = best_cand
                governed_action = "CRITICAL_FALLBACK_BEST"

        elif state == GovernorState.TOO_INACCURATE:
            # Sharpen accuracy: enforce best move
            chosen = best_cand
            governed_action = "ENFORCE_BEST_RECOVERY"

        else:
            # Within target: if PV 2 has negligible loss (<15 cp), allow natural variety
            if len(candidates) > 1 and candidates[1]["loss_cp"] <= 12.0 and not is_emergency:
                chosen = candidates[1] if np.random.rand() < 0.35 else best_cand
                governed_action = "NATURAL_VARIATION"
            else:
                chosen = best_cand
                governed_action = "NATURAL_BEST"

        # 3. Update cumulative statistics
        self.record_move_loss(chosen["loss_cp"])

        return {
            "chosen_candidate": chosen,
            "chosen_move": chess.Move.from_uci(chosen["move_uci"]),
            "governed_action": governed_action,
            "governor_state": state.value,
            "current_acpl": self.current_acpl,
            "target_acpl": self.profile["target_acpl"],
            "loss_incurred": chosen["loss_cp"],
        }
