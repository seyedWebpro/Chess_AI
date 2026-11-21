"""
dom_parsers.py: High-speed pure Python DOM element parsers for Chess.com and Lichess.
Transforms DOM element class names, styles, and data attributes into standard FEN positions.
"""

from typing import List, Dict, Optional, Tuple, Any
import re
from loguru import logger


# Chess.com piece code mapping: 'wn' -> 'N', 'bp' -> 'p'
CHESS_COM_PIECE_MAP = {
    "wp": "P", "wn": "N", "wb": "B", "wr": "R", "wq": "Q", "wk": "K",
    "bp": "p", "bn": "n", "bb": "b", "br": "r", "bq": "q", "bk": "k",
}

# Inverse mapping
CHAR_TO_CHESS_COM_PIECE = {v: k for k, v in CHESS_COM_PIECE_MAP.items()}

FILES = ["a", "b", "c", "d", "e", "f", "g", "h"]


def chess_com_square_to_san_square(sq_str: str) -> Optional[str]:
    """
    Converts Chess.com square identifier (e.g. 'square-52' or '52') to SAN square (e.g. 'e2').
    Square-XY: X is file (1-8 -> a-h), Y is rank (1-8).
    """
    match = re.search(r"(\d)(\d)", sq_str)
    if not match:
        return None
    f_idx = int(match.group(1)) - 1
    rank_idx = int(match.group(2))

    if 0 <= f_idx < 8 and 1 <= rank_idx <= 8:
        return f"{FILES[f_idx]}{rank_idx}"
    return None


def san_square_to_chess_com(san_sq: str) -> str:
    """Converts SAN square ('e2') to Chess.com format ('square-52')."""
    san = san_sq.lower().strip()
    f_idx = FILES.index(san[0]) + 1
    rank = san[1]
    return f"square-{f_idx}{rank}"


def infer_castling_rights(pieces: Dict[str, str]) -> str:
    """Infers plausible castling rights from king and rook positions."""
    rights = ""
    if pieces.get("e1") == "K":
        if pieces.get("h1") == "R":
            rights += "K"
        if pieces.get("a1") == "R":
            rights += "Q"
    if pieces.get("e8") == "k":
        if pieces.get("h8") == "r":
            rights += "k"
        if pieces.get("a8") == "r":
            rights += "q"
    return rights if rights else "-"


def pieces_dict_to_fen(
    pieces: Dict[str, str],
    active_color: str = "w",
    castling: str = "KQkq",
    en_passant: str = "-",
    halfmove: int = 0,
    fullmove: int = 1,
) -> str:
    """
    Converts a dictionary of {square_name: piece_char} into a full FEN string.
    Example: pieces={'e4': 'P', 'e1': 'K', ...}
    """
    ranks_fen = []
    for rank in range(8, 0, -1):
        empty_count = 0
        rank_str = ""
        for file_char in FILES:
            sq = f"{file_char}{rank}"
            piece = pieces.get(sq)
            if piece:
                if empty_count > 0:
                    rank_str += str(empty_count)
                    empty_count = 0
                rank_str += piece
            else:
                empty_count += 1
        if empty_count > 0:
            rank_str += str(empty_count)
        ranks_fen.append(rank_str)

    board_placement = "/".join(ranks_fen)
    return f"{board_placement} {active_color} {castling} {en_passant} {halfmove} {fullmove}"


def parse_chess_com_dom_pieces(
    class_list: List[str],
    active_color: str = "w",
    castling: str = "-",
    en_passant: str = "-",
    halfmove: int = 0,
    fullmove: int = 1,
) -> Tuple[str, Dict[str, str]]:
    """
    Parses class attribute strings from Chess.com DOM piece elements.
    Example class strings:
        ['piece wn square-52', 'piece bp square-47', 'piece wk square-51', ...]
    
    Returns:
        Tuple of (full_fen, pieces_dict)
    """
    pieces: Dict[str, str] = {}

    for cls_str in class_list:
        # Match piece type (e.g. wn, bp, wk)
        p_match = re.search(r"\b(w[pnbrqk]|b[pnbrqk])\b", cls_str)
        # Match square (e.g. square-52)
        sq_match = re.search(r"\bsquare-(\d\d)\b", cls_str)

        if p_match and sq_match:
            p_code = p_match.group(1)
            sq_id = sq_match.group(1)
            san_sq = chess_com_square_to_san_square(sq_id)
            piece_char = CHESS_COM_PIECE_MAP.get(p_code)

            if san_sq and piece_char:
                pieces[san_sq] = piece_char

    if castling == "auto":
        castling = infer_castling_rights(pieces)

    fen = pieces_dict_to_fen(
        pieces,
        active_color=active_color,
        castling=castling,
        en_passant=en_passant,
        halfmove=halfmove,
        fullmove=fullmove,
    )
    return fen, pieces


def extract_chess_com_highlights(class_list: List[str]) -> List[str]:
    """
    Extracts highlighted squares from Chess.com DOM elements.
    Elements usually have class like 'highlight square-52' or 'highlight square-54'.
    """
    squares: List[str] = []
    for cls_str in class_list:
        if "highlight" in cls_str:
            sq_match = re.search(r"\bsquare-(\d\d)\b", cls_str)
            if sq_match:
                san_sq = chess_com_square_to_san_square(sq_match.group(1))
                if san_sq and san_sq not in squares:
                    squares.append(san_sq)
    return squares


def parse_lichess_dom_pieces(
    piece_elements: List[Dict[str, Any]],
    active_color: str = "w",
    castling: str = "-",
    en_passant: str = "-",
    halfmove: int = 0,
    fullmove: int = 1,
) -> Tuple[str, Dict[str, str]]:
    """
    Parses Lichess DOM piece representations.
    Supports either:
    1. Direct square and class: {'square': 'e4', 'role': 'pawn', 'color': 'white'}
    2. Class string: {'class': 'white knight', 'square': 'g1'}
    3. Transform matrix / pixel offset: {'style': 'transform: translate(250px, 350px)', 'class': 'white knight'}
    """
    role_to_char = {
        ("white", "pawn"): "P", ("white", "knight"): "N", ("white", "bishop"): "B",
        ("white", "rook"): "R", ("white", "queen"): "Q", ("white", "king"): "K",
        ("black", "pawn"): "p", ("black", "knight"): "n", ("black", "bishop"): "b",
        ("black", "rook"): "r", ("black", "queen"): "q", ("black", "king"): "k",
    }

    pieces: Dict[str, str] = {}

    for elem in piece_elements:
        square = elem.get("square")
        color = elem.get("color")
        role = elem.get("role")

        # If role and color not directly provided, parse from 'class'
        if not (color and role) and "class" in elem:
            cls = elem["class"].lower()
            color = "white" if "white" in cls else ("black" if "black" in cls else None)
            for r in ["pawn", "knight", "bishop", "rook", "queen", "king"]:
                if r in cls:
                    role = r
                    break

        if square and color and role:
            char = role_to_char.get((color, role))
            if char:
                pieces[square.lower()] = char

    if castling == "auto":
        castling = infer_castling_rights(pieces)

    fen = pieces_dict_to_fen(
        pieces,
        active_color=active_color,
        castling=castling,
        en_passant=en_passant,
        halfmove=halfmove,
        fullmove=fullmove,
    )
    return fen, pieces
