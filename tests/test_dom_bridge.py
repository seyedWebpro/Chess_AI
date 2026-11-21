"""
tests/test_dom_bridge.py: Comprehensive Unit, Integration, and E2E Tests
for Phase 7: DOM Scraping, WebSocket Bridge, and Dual-Mode Input Router.
"""

import time
import json
import pytest
import asyncio
import chess
import numpy as np
import websockets
from playwright.sync_api import sync_playwright

from online.dom_parsers import (
    chess_com_square_to_san_square,
    san_square_to_chess_com,
    pieces_dict_to_fen,
    parse_chess_com_dom_pieces,
    parse_lichess_dom_pieces,
    extract_chess_com_highlights,
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
from screen.screen_board_detector import ScreenBoardDetector
from screen.screen_move_detector import ScreenMoveDetector
from tests.test_screen_cv import create_synthetic_screen_with_board


def test_01_chess_com_square_mapping():
    assert chess_com_square_to_san_square("square-11") == "a1"
    assert chess_com_square_to_san_square("square-88") == "h8"
    assert chess_com_square_to_san_square("square-52") == "e2"
    assert chess_com_square_to_san_square("square-54") == "e4"
    assert chess_com_square_to_san_square("invalid") is None

    assert san_square_to_chess_com("a1") == "square-11"
    assert san_square_to_chess_com("h8") == "square-88"
    assert san_square_to_chess_com("e4") == "square-54"


def test_02_chess_com_dom_piece_parsing_and_fen():
    # Initial board representation from Chess.com DOM classes
    class_list = [
        # White Rooks, Knights, Bishops, Queen, King
        "piece wr square-11", "piece wn square-21", "piece wb square-31", "piece wq square-41",
        "piece wk square-51", "piece wb square-61", "piece wn square-71", "piece wr square-81",
        # White Pawns
        "piece wp square-12", "piece wp square-22", "piece wp square-32", "piece wp square-42",
        "piece wp square-52", "piece wp square-62", "piece wp square-72", "piece wp square-82",
        # Black Pawns
        "piece bp square-17", "piece bp square-27", "piece bp square-37", "piece bp square-47",
        "piece bp square-57", "piece bp square-67", "piece bp square-77", "piece bp square-87",
        # Black Pieces
        "piece br square-18", "piece bn square-28", "piece bb square-38", "piece bq square-48",
        "piece bk square-58", "piece bb square-68", "piece bn square-78", "piece br square-88",
    ]

    fen, pieces = parse_chess_com_dom_pieces(class_list)
    assert len(pieces) == 32
    assert pieces["e1"] == "K"
    assert pieces["e8"] == "k"
    assert pieces["e2"] == "P"
    assert fen.startswith("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")


def test_03_extract_chess_com_highlights():
    classes = [
        "highlight square-52",
        "highlight square-54",
        "some-other-class",
    ]
    highlights = extract_chess_com_highlights(classes)
    assert "e2" in highlights
    assert "e4" in highlights
    assert len(highlights) == 2


def test_04_lichess_dom_piece_parsing():
    elements = [
        {"color": "white", "role": "king", "square": "e1"},
        {"color": "black", "role": "king", "square": "e8"},
        {"class": "white pawn", "square": "e4"},
    ]
    fen, pieces = parse_lichess_dom_pieces(elements)
    assert pieces["e1"] == "K"
    assert pieces["e8"] == "k"
    assert pieces["e4"] == "P"
    assert "4K3" in fen or "4P3" in fen


def test_05_websocket_bridge_server_lifecycle():
    server = ThreadedDOMBridgeServer(host="127.0.0.1", port=8768)
    server.start()
    assert server.client_count == 0

    async def connect_client():
        async with websockets.connect("ws://127.0.0.1:8768") as ws:
            await asyncio.sleep(0.05)
            # Send sample dom_update
            payload = {
                "type": "dom_update",
                "payload": {
                    "site": "chess.com",
                    "pieces": ["piece wk square-51", "piece bk square-58", "piece wp square-54"],
                    "highlights": ["highlight square-52", "highlight square-54"],
                    "orientation": "white",
                    "timestamp": time.time(),
                }
            }
            await ws.send(json.dumps(payload))
            await asyncio.sleep(0.05)

    asyncio.run(connect_client())

    time.sleep(0.1)
    state = server.latest_state
    assert state is not None
    assert state.site == "chess.com"
    assert "e4" in state.highlights
    assert "e2" in state.highlights
    assert state.pieces["e4"] == "P"

    server.stop()


def test_06_input_router_dom_only():
    router = InputRouter(mode=InputMode.DOM_ONLY)

    # Initial state
    assert len(router.board.move_stack) == 0

    # 1. White plays e4 (represented by piece placement and highlights)
    state1 = DOMGameState(
        fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        pieces={"e4": "P", "e1": "K", "e8": "k"},
        site="chess.com",
        orientation="white",
        highlights=["e2", "e4"],
        last_move="e2e4",
        timestamp=time.time(),
    )
    router.handle_dom_update(state1)

    assert router.last_move is not None
    assert router.last_move.uci() == "e2e4"
    assert router.last_san == "e4"
    assert router.active_source == ActiveInputSource.DOM

    # 2. Black plays e5
    state2 = DOMGameState(
        fen="rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        pieces={"e4": "P", "e5": "p", "e1": "K", "e8": "k"},
        site="chess.com",
        orientation="white",
        highlights=["e7", "e5"],
        last_move="e7e5",
        timestamp=time.time(),
    )
    router.handle_dom_update(state2)

    assert router.last_move.uci() == "e7e5"
    assert router.last_san == "e5"
    assert len(router.board.move_stack) == 2


def test_07_input_router_hybrid_failover_to_screen_cv():
    # Setup Screen CV Move Detector
    board_det = ScreenBoardDetector(board_bbox=(200, 100, 640, 640))
    screen_move_det = ScreenMoveDetector(board_detector=board_det)

    # Setup Router with short failover timeout (0.1s)
    router = InputRouter(
        mode=InputMode.HYBRID_AUTO,
        screen_move_detector=screen_move_det,
        failover_timeout_s=0.1,
    )

    # Initially DOM is silent -> Should use Screen CV
    desktop = create_synthetic_screen_with_board(
        board_x=200, board_y=100, board_size=640,
        highlight_squares={"e2": "yellow", "e4": "yellow"},
    )
    board_crop = desktop[100:740, 200:840]

    # Process frame via Screen CV
    res = router.process_screen_frame(board_crop)
    assert res is not None
    move, conf = res
    assert move.uci() == "e2e4"
    assert router.active_source == ActiveInputSource.SCREEN_CV

    # Now DOM wakes up and delivers Black move c7c5
    state_dom = DOMGameState(
        fen="rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
        pieces={"e4": "P", "c5": "p"},
        site="chess.com",
        orientation="white",
        highlights=["c7", "c5"],
        last_move="c7c5",
        timestamp=time.time(),
    )
    router.handle_dom_update(state_dom)

    # Router seamlessly switches active source to DOM
    assert router.active_source == ActiveInputSource.DOM
    assert router.last_move.uci() == "c7c5"
    assert len(router.board.move_stack) == 2


def test_08_playwright_e2e_browser_dom_bridge_integration():
    """
    Playwright E2E Test: Loads a synthetic Chess.com browser page,
    injects userscript, simulates live piece movement in DOM, and verifies
    instant reception by the Python WebSocket server and InputRouter.
    """
    server = ThreadedDOMBridgeServer(host="127.0.0.1", port=8769)
    server.start()

    router = InputRouter(mode=InputMode.DOM_ONLY, dom_server=server)

    html_content = """<!DOCTYPE html>
<html>
<head><title>Simulated Chess.com</title></head>
<body>
    <chess-board class="board">
        <div class="piece wr square-11"></div>
        <div class="piece wk square-51"></div>
        <div class="piece wp square-52"></div>
        <div class="piece bk square-58"></div>
        <div class="piece bp square-57"></div>
    </chess-board>
</body>
</html>"""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content(html_content)

        # Inject and start userscript bridge in the page
        script_code = """
        (function() {
            const ws = new WebSocket('ws://127.0.0.1:8769');
            window.testWs = ws;
            ws.onopen = () => {
                window.wsConnected = true;
                sendBoard();
            };
            function sendBoard() {
                const board = document.querySelector('chess-board');
                const pieces = [];
                board.querySelectorAll('.piece').forEach(p => pieces.push(p.className));
                const highlights = [];
                board.querySelectorAll('.highlight').forEach(h => highlights.push(h.className));
                ws.send(JSON.stringify({
                    type: 'dom_update',
                    payload: {
                        site: 'chess.com',
                        pieces: pieces,
                        highlights: highlights,
                        orientation: 'white',
                        timestamp: Date.now() / 1000.0
                    }
                }));
            }
            window.sendBoard = sendBoard;
        })();
        """
        page.evaluate(script_code)

        # Wait until client connects to Python server
        for _ in range(30):
            if server.client_count > 0:
                break
            time.sleep(0.05)

        assert server.client_count == 1, "Playwright browser tab should be connected to DOM Bridge."

        # Simulate move in DOM: e2 -> e4
        move_script = """
        const board = document.querySelector('chess-board');
        // Move pawn from square-52 to square-54
        const pawn = board.querySelector('.piece.wp.square-52');
        pawn.className = 'piece wp square-54';
        
        // Add highlights
        const h1 = document.createElement('div');
        h1.className = 'highlight square-52';
        const h2 = document.createElement('div');
        h2.className = 'highlight square-54';
        board.appendChild(h1);
        board.appendChild(h2);
        
        window.sendBoard();
        """
        page.evaluate(move_script)

        # Wait for router to process move
        for _ in range(30):
            if router.last_move is not None:
                break
            time.sleep(0.05)

        assert router.last_move is not None
        assert router.last_move.uci() == "e2e4"
        assert router.active_source == ActiveInputSource.DOM

        browser.close()

    server.stop()
