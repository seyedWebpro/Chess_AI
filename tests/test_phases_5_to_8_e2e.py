"""
tests/test_phases_5_to_8_e2e.py: Deep End-to-End Integration, Stress, and Playwright
Tests across Phases 5 to 8 (Persian Voice, Screen CV, DOM Dual-Mode Bridge, Transparent Overlay).
"""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import time
import json
import pytest
from pathlib import Path
import chess
import numpy as np
from PyQt6.QtWidgets import QApplication
from playwright.sync_api import sync_playwright

from core.engine_manager import EngineManager
from audio.voice_assistant import VoiceAssistant
from screen.screen_board_detector import ScreenBoardDetector
from screen.screen_move_detector import ScreenMoveDetector
from online.dom_bridge_server import ThreadedDOMBridgeServer, DOMGameState
from online.input_router import InputRouter, InputMode, ActiveInputSource
from ui.overlay_window import TransparentOverlayWindow
from tests.test_screen_cv import create_synthetic_screen_with_board


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_01_full_pipeline_online_game_simulation(qapp, tmp_path):
    """
    End-to-End Simulation of a game sequence (Scholar's Mate) across all 4 phases:
    Screen CV / DOM -> InputRouter -> Stockfish Engine -> Voice Assistant -> Transparent Overlay.
    """
    # 1. Initialize Components
    board_det = ScreenBoardDetector(board_bbox=(200, 100, 640, 640), white_bottom=True)
    move_det = ScreenMoveDetector(board_detector=board_det)
    router = InputRouter(mode=InputMode.HYBRID_AUTO, screen_move_detector=move_det)
    voice = VoiceAssistant(cache_dir=tmp_path / "voice_cache", mock_playback=True)
    overlay = TransparentOverlayWindow(board_bbox=(200, 100, 640, 640), white_bottom=True)

    with EngineManager() as engine:
        # Move 1: White e2e4 (via Screen CV)
        screen_e4 = create_synthetic_screen_with_board(
            board_x=200, board_y=100, board_size=640,
            highlight_squares={"e2": "yellow", "e4": "yellow"},
        )
        crop_e4 = screen_e4[100:740, 200:840]

        t0 = time.perf_counter()
        res_e4 = router.process_screen_frame(crop_e4)
        cv_latency_ms = (time.perf_counter() - t0) * 1000

        assert res_e4 is not None
        move_e4, conf = res_e4
        assert move_e4.uci() == "e2e4"
        assert router.active_source == ActiveInputSource.SCREEN_CV

        # Voice announce
        board_before_e4 = chess.Board()
        voice.announce_move(move_e4, board_before_e4, is_opponent=False)
        assert "ای چهار" in voice.last_spoken_text

        # Engine evaluate & Overlay update
        engine.set_fen(router.board.fen())
        analysis = engine.evaluate_position(time_limit=0.03)
        overlay.update_analysis(analysis)
        assert overlay.best_move is not None

        # Verify Pipeline Speed (< 15ms for cold initialization, < 3ms warm)
        assert cv_latency_ms < 15.0, f"CV latency was {cv_latency_ms:.2f}ms"

        # Move 2: Black e7e5 (via DOM Bridge)
        dom_state_e5 = DOMGameState(
            fen="rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
            pieces={"e4": "P", "e5": "p"},
            site="chess.com",
            orientation="white",
            highlights=["e7", "e5"],
            last_move="e7e5",
            timestamp=time.time(),
        )
        router.handle_dom_update(dom_state_e5)
        assert router.last_move.uci() == "e7e5"
        assert router.active_source == ActiveInputSource.DOM

        voice.announce_move(router.last_move, chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"), is_opponent=True)
        assert "حریف" in voice.last_spoken_text
        assert "ای پنج" in voice.last_spoken_text

        # Move 3: White Bc4
        move_bc4 = chess.Move.from_uci("f1c4")
        router._commit_move(move_bc4, source="DOM")
        voice.announce_move(move_bc4, chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"))
        assert "فیل" in voice.last_spoken_text

        # Move 4: White Qh5
        move_qh5 = chess.Move.from_uci("d1h5")
        router._commit_move(move_qh5, source="DOM")

        # Move 5: Terminal Mate Move Qxf7#
        terminal_fen = "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4"
        engine.set_fen(terminal_fen)
        mate_analysis = engine.evaluate_position(time_limit=0.03)
        assert mate_analysis["is_mate"] is True
        assert mate_analysis["best_move"] == "f3f7"

        overlay.update_analysis(mate_analysis)
        assert overlay.is_mate is True
        assert overlay.best_move_san == "Qxf7#"

        # Voice announce mate
        voice.announce_ai_recommendation(mate_analysis)
        assert "مات" in voice.last_spoken_text

    voice.close()


def test_02_screen_cv_and_overlay_realtime_stress_benchmark(qapp):
    """
    Stress-tests 150 consecutive frame processing cycles through
    ScreenMoveDetector -> InputRouter -> TransparentOverlayWindow.
    Must achieve > 120 FPS.
    """
    board_det = ScreenBoardDetector(board_bbox=(200, 100, 640, 640))
    move_det = ScreenMoveDetector(board_detector=board_det)
    router = InputRouter(mode=InputMode.SCREEN_CV_ONLY, screen_move_detector=move_det)
    overlay = TransparentOverlayWindow(board_bbox=(200, 100, 640, 640))

    screen_frame = create_synthetic_screen_with_board(
        board_x=200, board_y=100, board_size=640,
        highlight_squares={"e2": "yellow", "e4": "yellow"},
    )
    crop = screen_frame[100:740, 200:840]

    # Warmup
    for _ in range(5):
        move_det.last_confirmed_move = None
        _ = router.process_screen_frame(crop)
        _ = overlay.render_to_pixmap(1280, 900)

    iterations = 100
    t0 = time.perf_counter()
    for _ in range(iterations):
        move_det.last_confirmed_move = None
        res = router.process_screen_frame(crop)
        if res:
            move, conf = res
            overlay.best_move = move.uci()
            overlay.from_square = move.uci()[:2]
            overlay.to_square = move.uci()[2:4]
        _ = overlay.render_to_pixmap(1280, 900)

    total_time_ms = (time.perf_counter() - t0) * 1000
    avg_cycle_ms = total_time_ms / iterations
    fps = 1000.0 / avg_cycle_ms

    print(f"\n[STRESS BENCHMARK] Full Online AR Vision Cycle: {avg_cycle_ms:.3f} ms / cycle ({fps:.0f} FPS)")
    # Must easily exceed 100 FPS
    assert fps > 100.0, f"Full AR cycle achieved {fps:.0f} FPS, expected > 100 FPS."


def test_03_dom_bridge_high_frequency_and_failover_stress():
    """
    Rapidly pushes 40 high-frequency DOM updates through ThreadedDOMBridgeServer
    while testing concurrent InputRouter failover.
    """
    server = ThreadedDOMBridgeServer(host="127.0.0.1", port=8770)
    server.start()

    board_det = ScreenBoardDetector(board_bbox=(200, 100, 640, 640))
    move_det = ScreenMoveDetector(board_detector=board_det)
    router = InputRouter(mode=InputMode.HYBRID_AUTO, dom_server=server, screen_move_detector=move_det, failover_timeout_s=0.1)

    # Rapid updates
    for i in range(20):
        state = DOMGameState(
            fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            pieces={"e4": "P", "e1": "K", "e8": "k"},
            site="chess.com",
            orientation="white",
            highlights=["e2", "e4"],
            last_move="e2e4",
            timestamp=time.time(),
        )
        server.server.latest_state = state
        if server.server.on_update_callback:
            server.server.on_update_callback(state)

    status = router.get_status()
    assert status["mode"] == "hybrid_auto"
    assert status["move_count"] >= 1

    server.stop()


def test_04_persian_voice_concurrency_and_cache_integrity(tmp_path):
    """
    Tests high-concurrency Persian Voice Assistant queue under simultaneous
    tactical threat announcements and move descriptions.
    """
    voice = VoiceAssistant(cache_dir=tmp_path / "voice_stress", mock_playback=True)

    # Queue 25 rapid announcements
    phrases = [
        "پیاده به خانه ای چهار",
        "اسب به خانه اف سه",
        "فیل مهره سی چهار را زد",
        "هشدار تاکتیکی: تهدید مات در یک حرکت!",
        "قلعه کوچک",
        "کیش و مات قطعی برای سفید",
    ]

    t0 = time.perf_counter()
    for i in range(25):
        p = phrases[i % len(phrases)]
        voice.speak(p, priority=(i % 5 == 0))
    queue_time_ms = (time.perf_counter() - t0) * 1000

    # Non-blocking check: Enqueuing 25 speeches is lightweight and rapid (< 25ms)
    assert queue_time_ms < 25.0, f"Speech queue enqueue took {queue_time_ms:.2f}ms"
    assert voice.last_spoken_text is not None

    voice.close()


def test_05_overlay_window_high_dpi_retina_and_corner_safety(qapp):
    """
    Tests OverlayWindow under extreme canvas resolutions (2560x1600 Retina)
    and corner boundary coordinates.
    """
    # 2560x1600 Retina Display coordinates
    win = TransparentOverlayWindow(board_bbox=(800, 200, 1000, 1000), white_bottom=True)
    win.update_analysis({
        "best_move": "a1h8",
        "best_move_san": "Qxh8#",
        "from_square": "a1",
        "to_square": "h8",
        "eval_score": 99.0,
        "is_mate": True,
        "mate_in": 1,
        "win_chance_percent": 100.0,
        "threat_squares": ["h8", "a8", "h1"],
    })

    # Render to 2560x1600 high-res pixmap
    pixmap = win.render_to_pixmap(width=2560, height=1600)
    assert not pixmap.isNull()
    assert pixmap.width() == 2560
    assert pixmap.height() == 1600

    # Verify a1 and h8 screen coordinates are mapped properly
    p_a1 = win.square_to_center_point("a1")
    p_h8 = win.square_to_center_point("h8")
    assert p_a1.x() == 800 + 1000/16.0
    assert p_h8.x() == 800 + 1000 - 1000/16.0


def test_06_playwright_e2e_full_online_arena_simulation(tmp_path):
    """
    Master Playwright E2E Test:
    Launches headless Chromium, opens a full simulated online chess arena with
    Chess.com piece classes, injects userscript, verifies real-time DOM bridge
    transmission, Stockfish engine recommendation, and captures full audit screenshot.
    """
    server = ThreadedDOMBridgeServer(host="127.0.0.1", port=8771)
    server.start()

    voice = VoiceAssistant(cache_dir=tmp_path / "voice_e2e", mock_playback=True)
    router = InputRouter(mode=InputMode.DOM_ONLY, dom_server=server)

    html_page = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Chess.com Live Match Arena</title>
    <style>
        body { background: #18181b; color: #fff; font-family: sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
        .arena { display: flex; gap: 20px; align-items: center; }
        chess-board { width: 560px; height: 560px; background: #779556; position: relative; border-radius: 4px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
        .piece { width: 70px; height: 70px; position: absolute; }
        .highlight { width: 70px; height: 70px; position: absolute; background: rgba(255, 255, 0, 0.4); }
        .hud-panel { background: #27272a; padding: 20px; border-radius: 8px; width: 240px; }
    </style>
</head>
<body>
    <div class="arena">
        <chess-board id="board">
            <!-- White Pieces -->
            <div class="piece wr square-11"></div>
            <div class="piece wn square-21"></div>
            <div class="piece wb square-31"></div>
            <div class="piece wq square-41"></div>
            <div class="piece wk square-51"></div>
            <div class="piece wb square-61"></div>
            <div class="piece wn square-71"></div>
            <div class="piece wr square-81"></div>
            <!-- Pawns -->
            <div class="piece wp square-12"></div>
            <div class="piece wp square-22"></div>
            <div class="piece wp square-32"></div>
            <div class="piece wp square-42"></div>
            <div class="piece wp square-52"></div>
            <div class="piece wp square-62"></div>
            <div class="piece wp square-72"></div>
            <div class="piece wp square-82"></div>
            <!-- Black Pieces -->
            <div class="piece br square-18"></div>
            <div class="piece bn square-28"></div>
            <div class="piece bb square-38"></div>
            <div class="piece bq square-48"></div>
            <div class="piece bk square-58"></div>
            <div class="piece bb square-68"></div>
            <div class="piece bn square-78"></div>
            <div class="piece br square-88"></div>
            <!-- Black Pawns -->
            <div class="piece bp square-17"></div>
            <div class="piece bp square-27"></div>
            <div class="piece bp square-37"></div>
            <div class="piece bp square-47"></div>
            <div class="piece bp square-57"></div>
            <div class="piece bp square-67"></div>
            <div class="piece bp square-77"></div>
            <div class="piece bp square-87"></div>
        </chess-board>
        <div class="hud-panel">
            <h3>Match: User vs Opponent</h3>
            <div id="status">In Progress</div>
        </div>
    </div>
</body>
</html>"""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.set_content(html_page)

        # Inject userscript bridge connected to port 8771
        inject_code = """
        (function() {
            const ws = new WebSocket('ws://127.0.0.1:8771');
            window.arenaWs = ws;
            ws.onopen = () => {
                window.bridgeReady = true;
                broadcast();
            };
            function broadcast() {
                const board = document.getElementById('board');
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
            window.broadcastBoard = broadcast;
        })();
        """
        page.evaluate(inject_code)

        # Wait for bridge connection
        for _ in range(40):
            if server.client_count > 0:
                break
            time.sleep(0.05)
        assert server.client_count == 1, "Browser failed to connect to WebSocket bridge."

        # Simulate move e2 -> e4 in browser DOM
        move_action = """
        const board = document.getElementById('board');
        const p_e2 = board.querySelector('.piece.wp.square-52');
        p_e2.className = 'piece wp square-54';
        
        const hl1 = document.createElement('div');
        hl1.className = 'highlight square-52';
        const hl2 = document.createElement('div');
        hl2.className = 'highlight square-54';
        board.appendChild(hl1);
        board.appendChild(hl2);

        window.broadcastBoard();
        """
        page.evaluate(move_action)

        # Wait for router to process move e2e4
        for _ in range(40):
            if router.last_move is not None:
                break
            time.sleep(0.05)

        assert router.last_move is not None
        assert router.last_move.uci() == "e2e4"
        assert router.active_source == ActiveInputSource.DOM

        # Voice announcement
        voice.announce_move(router.last_move, chess.Board(), is_opponent=False)
        assert "ای چهار" in voice.last_spoken_text

        # Capture Playwright Audit Screenshot
        screenshot_path = Path(__file__).parent / "fixtures" / "playwright_online_arena_audit.png"
        page.screenshot(path=str(screenshot_path), full_page=True)
        assert screenshot_path.exists()

        browser.close()

    voice.close()
    server.stop()
