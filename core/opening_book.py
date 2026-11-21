"""
opening_book.py: Polyglot Opening Book Manager & Generator.
Loads, queries, and builds standard 16-byte binary Polyglot opening books (.bin)
with 64-bit Zobrist hashing for instant, zero-latency opening play.
"""

from typing import List, Tuple, Optional, Dict
from pathlib import Path
import struct
import random
import chess
import chess.polyglot
from loguru import logger


# Standard Master Opening Lines for default built-in Polyglot book
STANDARD_OPENINGS: Dict[str, List[str]] = {
    "Ruy Lopez (Spanish)": [
        "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7", "f1e1", "b7b5", "a4b3", "d7d6"
    ],
    "Italian Game (Giuoco Piano)": [
        "e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3", "g8f6", "d2d3", "d7d6", "e1g1", "e8g8"
    ],
    "Sicilian Defense (Open / Najdorf)": [
        "e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6", "c1e3", "e7e5"
    ],
    "Sicilian Defense (Dragon)": [
        "e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "g7g6", "c1e3", "f8g7"
    ],
    "Queen's Gambit Declined": [
        "d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7", "e2e3", "e8g8", "g1f3", "b8d7"
    ],
    "Queen's Gambit Accepted": [
        "d2d4", "d7d5", "c2c4", "d5c4", "g1f3", "g8f6", "e2e3", "e7e6", "f1c4", "c7c5"
    ],
    "French Defense (Winawer / Classical)": [
        "e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "f8b4", "e4e5", "c7c5", "a2a3", "b4c3", "b2c3"
    ],
    "Caro-Kann Defense (Classical)": [
        "e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4", "c8f5", "e4g3", "f5g6", "h2h4", "h7h6"
    ],
    "King's Indian Defense": [
        "d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6", "g1f3", "e8g8", "f1e2", "e7e5"
    ],
    "English Opening": [
        "c2c4", "e7e5", "b1c3", "g8f6", "g1f3", "b8c6", "g2g3", "d7d5", "c4d5", "f3d5"
    ],
    "Scandinavian Defense": [
        "e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5", "d2d4", "g8f6", "g1f3", "c7c6"
    ],
    "Nimzo-Indian Defense": [
        "d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "e2e3", "O-O", "f1d3", "d7d5"
    ],
}


def encode_polyglot_move(move: chess.Move) -> int:
    """
    Encodes a python-chess Move into a Polyglot 16-bit integer format:
    bits 0-2: to_file (0-7)
    bits 3-5: to_rank (0-7)
    bits 6-8: from_file (0-7)
    bits 9-11: from_rank (0-7)
    bits 12-14: promotion (0: none, 1: knight, 2: bishop, 3: rook, 4: queen)
    """
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    from_file = chess.square_file(move.from_square)
    from_rank = chess.square_rank(move.from_square)

    promo_map = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}
    promo_code = promo_map.get(move.promotion, 0)

    return (promo_code << 12) | (from_rank << 9) | (from_file << 6) | (to_rank << 3) | to_file


def build_polyglot_binary_file(output_path: Path, openings: Dict[str, List[str]] = STANDARD_OPENINGS) -> int:
    """
    Constructs a binary Polyglot .bin file from opening move sequences.
    Ensures all entries are sorted ascending by 64-bit Zobrist key as required by Polyglot spec.
    """
    entries: List[Tuple[int, int, int, int]] = []  # (key, raw_move, weight, learn)
    seen = set()

    for name, line in openings.items():
        board = chess.Board()
        weight = 300  # High initial weight for primary book moves
        for uci in line:
            # Handle castling UCI 'O-O' or standard 'e1g1'
            try:
                move = chess.Move.from_uci(uci) if uci in [m.uci() for m in board.legal_moves] else board.parse_san(uci)
            except Exception:
                continue

            if move not in board.legal_moves:
                break

            key = chess.polyglot.zobrist_hash(board)
            raw_move = encode_polyglot_move(move)

            # Polyglot castling moves are encoded as king takes own rook in standard spec
            if board.is_castling(move):
                # Kingside: e1h1 (white) / e8h8 (black)
                # Queenside: e1a1 (white) / e8a8 (black)
                if move.to_square > move.from_square:  # Kingside
                    rook_sq = chess.H1 if board.turn == chess.WHITE else chess.H8
                else:  # Queenside
                    rook_sq = chess.A1 if board.turn == chess.WHITE else chess.A8
                castling_move = chess.Move(move.from_square, rook_sq)
                raw_move = encode_polyglot_move(castling_move)

            entry_id = (key, raw_move)
            if entry_id not in seen:
                seen.add(entry_id)
                entries.append((key, raw_move, weight, 0))

            board.push(move)
            weight = max(50, weight - 18)

    # Sort strictly by 64-bit Zobrist key ascending
    entries.sort(key=lambda e: (e[0], -e[2]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        for key, raw_move, weight, learn in entries:
            f.write(chess.polyglot.ENTRY_STRUCT.pack(key, raw_move, weight, learn))

    logger.info(f"Built Polyglot opening book at {output_path} with {len(entries)} entries.")
    return len(entries)


class PolyglotBookManager:
    """
    Manages opening book lookup and querying using python-chess Polyglot reader.
    """

    def __init__(self, book_path: Optional[Path] = None):
        self.book_path = book_path or Path(__file__).parent.parent / "data" / "polyglot" / "master_book.bin"

        if not self.book_path.exists():
            logger.warning(f"Opening book not found at {self.book_path}. Building default book...")
            build_polyglot_binary_file(self.book_path)

        self._reader: Optional[chess.polyglot.MemoryMappedReader] = None
        self._load_reader()

    def _load_reader(self) -> None:
        try:
            self._reader = chess.polyglot.open_reader(str(self.book_path))
            logger.success(f"Polyglot opening book opened from: {self.book_path}")
        except Exception as e:
            logger.error(f"Failed to open Polyglot book: {e}")
            self._reader = None

    def is_in_book(self, board: chess.Board) -> bool:
        """Returns True if current board position exists in opening book."""
        if not self._reader:
            return False
        try:
            return self._reader.find(board) is not None
        except IndexError:
            return False

    def get_all_book_moves(self, board: chess.Board) -> List[Tuple[chess.Move, int]]:
        """Returns all available book moves with their respective weights."""
        if not self._reader:
            return []
        moves = []
        try:
            for entry in self._reader.find_all(board):
                moves.append((entry.move, entry.weight))
        except Exception:
            pass
        return moves

    def get_book_move(
        self,
        board: chess.Board,
        mode: str = "weighted",  # "weighted", "top", or "random"
    ) -> Optional[chess.Move]:
        """
        Retrieves a book move for the given board position.
        
        Args:
            board: Current chess.Board.
            mode: "weighted" (roulette selection by weight), "top" (highest weight), "random".
            
        Returns:
            chess.Move or None if out of book.
        """
        all_moves = self.get_all_book_moves(board)
        if not all_moves:
            return None

        if mode == "top":
            # Highest weight move
            return max(all_moves, key=lambda x: x[1])[0]

        if mode == "random":
            return random.choice(all_moves)[0]

        # Weighted probability selection
        moves, weights = zip(*all_moves)
        total_w = sum(weights)
        if total_w <= 0:
            return moves[0]

        return random.choices(moves, weights=weights, k=1)[0]

    def close(self) -> None:
        if self._reader:
            try:
                self._reader.close()
            except Exception:
                pass
            self._reader = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
