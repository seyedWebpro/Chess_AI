"""
Online package: Dual Mode Chess Input (DOM Scraping & Screen CV Bridge).
"""

from online.dom_parsers import (
    parse_chess_com_dom_pieces,
    parse_lichess_dom_pieces,
    pieces_dict_to_fen,
    chess_com_square_to_san_square,
)
from online.dom_bridge_server import (
    DOMBridgeServer,
    ThreadedDOMBridgeServer,
    DOMGameState,
)
from online.input_router import (
    InputRouter,
    InputMode,
    ActiveInputSource,
)

__all__ = [
    "parse_chess_com_dom_pieces",
    "parse_lichess_dom_pieces",
    "pieces_dict_to_fen",
    "chess_com_square_to_san_square",
    "DOMBridgeServer",
    "ThreadedDOMBridgeServer",
    "DOMGameState",
    "InputRouter",
    "InputMode",
    "ActiveInputSource",
]
