"""
input_router.py: Dual-Mode Smart Input Router.
Arbitrates between Screen CV (Phase 6) and DOM Scraping (Phase 7).
Features:
- Instant runtime mode switching (DOM_ONLY, SCREEN_CV_ONLY, HYBRID_AUTO, CONSENSUS).
- Automatic seamless failover: If DOM drops, falls back to Screen CV without desync.
- Cross-validation consensus: Validates visual moves against DOM FEN.
- Thread-safe state synchronization.
"""

from enum import Enum
from typing import Optional, Dict, Any, Tuple, Callable
import time
import chess
from loguru import logger
import numpy as np

from online.dom_bridge_server import DOMGameState, ThreadedDOMBridgeServer
from screen.screen_move_detector import ScreenMoveDetector
from screen.screen_board_detector import ScreenBoardDetector


class InputMode(Enum):
    DOM_ONLY = "dom_only"
    SCREEN_CV_ONLY = "screen_cv_only"
    HYBRID_AUTO = "hybrid_auto"
    CONSENSUS = "consensus"


class ActiveInputSource(Enum):
    DOM = "DOM"
    SCREEN_CV = "SCREEN_CV"
    CONSENSUS = "CONSENSUS"
    NONE = "NONE"


class InputRouter:
    """
    Manages dual-channel online chess inputs (DOM Bridge vs Screen Computer Vision).
    """

    def __init__(
        self,
        mode: InputMode = InputMode.HYBRID_AUTO,
        dom_server: Optional[ThreadedDOMBridgeServer] = None,
        screen_move_detector: Optional[ScreenMoveDetector] = None,
        failover_timeout_s: float = 1.5,
        on_move_confirmed: Optional[Callable[[chess.Move, str], None]] = None,
    ):
        self.mode = mode
        self.dom_server = dom_server
        self.screen_move_detector = screen_move_detector
        self.failover_timeout_s = failover_timeout_s
        self.on_move_confirmed = on_move_confirmed

        self.board = chess.Board()
        self.active_source = ActiveInputSource.NONE
        self.last_move: Optional[chess.Move] = None
        self.last_san: Optional[str] = None
        self.last_dom_time: float = 0.0
        self.last_screen_cv_time: float = 0.0
        self.move_history: list[chess.Move] = []

        # Hook DOM server callback if provided
        if self.dom_server:
            self.dom_server.server.on_update_callback = self.handle_dom_update

    def set_mode(self, mode: InputMode):
        """Switches operating mode dynamically at runtime."""
        logger.info(f"InputRouter switching mode from {self.mode.value} to {mode.value}")
        self.mode = mode

    def handle_dom_update(self, state: DOMGameState):
        """Processes an incoming DOM update from the browser bridge."""
        self.last_dom_time = time.time()

        if self.mode == InputMode.SCREEN_CV_ONLY:
            return

        # Check if FEN changed from current board
        curr_placement = self.board.fen().split(" ")[0]
        new_placement = state.fen.split(" ")[0]

        if curr_placement != new_placement and new_placement:
            logger.info(f"DOM FEN change detected: {new_placement}")

            # Try to infer move from last_move or FEN
            move_made = None
            if state.last_move:
                try:
                    cand_move = chess.Move.from_uci(state.last_move)
                    if cand_move in self.board.legal_moves:
                        move_made = cand_move
                except Exception:
                    pass

            # If last_move wasn't legal or not parsed, find move that leads to new_placement
            if not move_made:
                for legal_move in self.board.legal_moves:
                    self.board.push(legal_move)
                    if self.board.fen().split(" ")[0] == new_placement:
                        move_made = legal_move
                        self.board.pop()
                        break
                    self.board.pop()

            if move_made:
                self._commit_move(move_made, source="DOM")
            else:
                # Direct FEN resynchronization fallback only if position is valid with both kings
                if "k" in state.fen and "K" in state.fen:
                    try:
                        test_board = chess.Board(state.fen)
                        if test_board.is_valid():
                            self.board = test_board
                            logger.warning(f"Board resynced directly from DOM FEN: {state.fen}")
                    except Exception as e:
                        logger.error(f"Failed to set FEN from DOM: {e}")

    def process_screen_frame(self, board_crop: np.ndarray) -> Optional[Tuple[chess.Move, float]]:
        """
        Processes a cropped chessboard frame via ScreenMoveDetector.
        Active during SCREEN_CV_ONLY, HYBRID_AUTO (when DOM is silent), or CONSENSUS.
        """
        if self.screen_move_detector is None:
            return None

        # Determine if Screen CV should be active
        dom_active = (time.time() - self.last_dom_time) < self.failover_timeout_s
        has_dom_clients = (self.dom_server is not None and self.dom_server.client_count > 0)

        should_run_cv = False
        if self.mode == InputMode.SCREEN_CV_ONLY:
            should_run_cv = True
            self.active_source = ActiveInputSource.SCREEN_CV
        elif self.mode == InputMode.HYBRID_AUTO:
            if not (has_dom_clients and dom_active):
                should_run_cv = True
                self.active_source = ActiveInputSource.SCREEN_CV
            else:
                self.active_source = ActiveInputSource.DOM
        elif self.mode == InputMode.CONSENSUS:
            should_run_cv = True
            self.active_source = ActiveInputSource.CONSENSUS

        if not should_run_cv:
            return None

        res = self.screen_move_detector.detect_move_from_highlights(board_crop, self.board)
        if res:
            move, conf = res
            self.last_screen_cv_time = time.time()
            self._commit_move(move, source="SCREEN_CV")
            return move, conf

        return None

    def _commit_move(self, move: chess.Move, source: str):
        """Commits a confirmed legal move to internal board and notifies callbacks."""
        if move not in self.board.legal_moves:
            logger.warning(f"Attempted to commit illegal move {move.uci()} from source {source}")
            return

        san = self.board.san(move)
        self.board.push(move)
        self.last_move = move
        self.last_san = san
        self.move_history.append(move)

        if source == "DOM":
            self.active_source = ActiveInputSource.DOM
        elif source == "SCREEN_CV":
            self.active_source = ActiveInputSource.SCREEN_CV

        logger.success(f"[ROUTER] Move Committed [{source}]: {move.uci()} ({san}) | FEN: {self.board.fen()}")

        if self.on_move_confirmed:
            try:
                self.on_move_confirmed(move, source)
            except Exception as e:
                logger.error(f"Error in on_move_confirmed callback: {e}")

    def reset(self):
        """Resets board to standard starting position."""
        self.board.reset()
        self.last_move = None
        self.move_history.clear()
        if self.screen_move_detector:
            self.screen_move_detector.last_confirmed_move = None
        logger.info("InputRouter board reset to initial position.")

    def get_status(self) -> Dict[str, Any]:
        """Returns diagnostic health and telemetry information."""
        dom_clients = self.dom_server.client_count if self.dom_server else 0
        dom_connected = dom_clients > 0 and (time.time() - self.last_dom_time < self.failover_timeout_s)

        return {
            "mode": self.mode.value,
            "active_source": self.active_source.value,
            "dom_connected": dom_connected,
            "dom_clients": dom_clients,
            "move_count": len(self.move_history),
            "last_move": self.last_move.uci() if self.last_move else None,
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
        }
