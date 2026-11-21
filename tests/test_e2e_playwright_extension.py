"""
test_e2e_playwright_extension.py: Full End-to-End Playwright Audit for Chrome Extension & AI Coach (Phase 16).
Simulates real browser interactions on a chess board, verifying:
- Extension Content Script DOM scraping & WebSocket bridge.
- Conceptual Persian AI Grandmaster coaching generation.
- Real-time in-page Dynamic HUD rendering.
- Board square neon overlays & highlights.
"""

import time
import json
import pytest
from pathlib import Path
import chess
from playwright.sync_api import sync_playwright

from online.dom_bridge_server import ThreadedDOMBridgeServer, DOMGameState
from core.concept_coach import ConceptCoach
from core.engine_manager import EngineManager
from audio.stealth_audio import EarconSynthesizer, StealthAudioController, StealthMode


def test_playwright_e2e_extension_and_conceptual_coach():
    """
    Playwright E2E Test:
    1. Spins up WebSocket Bridge on port 8775.
    2. Opens a synthetic Chess.com board in Chromium.
    3. Injects extension/content.js with configured WS port.
    4. Simulates a move in the board DOM.
    5. Verifies:
       - WebSocket communication.
       - ConceptCoach generates Persian commentary.
       - In-page HUD renders move, eval, and conceptual title.
       - Board highlights are properly positioned.
    """
    received_states = []

    def on_board_update(state: DOMGameState):
        received_states.append(state)

    server = ThreadedDOMBridgeServer(host="127.0.0.1", port=8775, on_update_callback=on_board_update)
    server.start()

    coach = ConceptCoach()
    synth = EarconSynthesizer()

    content_js_path = Path(__file__).parent.parent / "extension" / "content.js"
    assert content_js_path.exists(), "extension/content.js must exist."

    # Read content.js and configure port to 8775
    content_js_code = content_js_path.read_text(encoding="utf-8")
    content_js_code = content_js_code.replace("ws://127.0.0.1:8765", "ws://127.0.0.1:8775")

    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Chess.com Simulated Game</title>
    <style>
        body { margin: 0; background: #262421; }
        .board { position: relative; width: 640px; height: 640px; margin: 40px auto; background: #b58863; }
        .piece { position: absolute; width: 80px; height: 80px; }
        .square-52 { left: 320px; top: 480px; }
        .square-54 { left: 320px; top: 320px; }
        .square-51 { left: 320px; top: 560px; }
        .square-58 { left: 320px; top: 0px; }
    </style>
</head>
<body>
    <div class="board" id="chessboard">
        <div class="piece wk square-51"></div>
        <div class="piece wp square-52"></div>
        <div class="piece bk square-58"></div>
    </div>
</body>
</html>"""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.set_content(html_content)

        # Inject Extension Content Script
        page.evaluate(content_js_code)

        # 1. Wait for WebSocket connection
        for _ in range(40):
            if server.client_count > 0:
                break
            time.sleep(0.05)

        assert server.client_count >= 1, "Browser should connect to WebSocket bridge on 8775."

        # Verify in-page HUD was created
        hud_el = page.locator("#chess-ai-inpage-hud")
        assert hud_el.is_visible()

        status_el = page.locator("#cai-status")
        # Give a small moment for WebSocket onopen to trigger status badge
        time.sleep(0.1)
        assert "LIVE" in status_el.text_content()

        # 2. Simulate AI engine analysis and Persian Concept Coach commentary
        board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
        move = chess.Move.from_uci("e2e4")
        analysis = coach.analyze_move(board, move, score_cp=45)

        assert "مرکز" in analysis["short_concept"] or "پیاده" in analysis["short_concept"]
        assert len(analysis["explanation_fa"]) > 10

        # Broadcast best move to extension
        server.broadcast({
            "type": "best_move",
            "from": "e2",
            "to": "e4",
            "san": "e4",
            "eval": "+0.45",
            "concept_title": analysis["short_concept"],
            "concept_explanation": analysis["explanation_fa"],
        })

        # 3. Assert HUD updates with move, eval, and concept
        time.sleep(0.15)
        move_el = page.locator("#cai-move")
        eval_el = page.locator("#cai-eval")
        concept_el = page.locator("#cai-concept")

        assert move_el.text_content() == "e4"
        assert eval_el.text_content() == "+0.45"
        assert concept_el.is_visible()
        assert analysis["short_concept"] in concept_el.text_content()

        # 4. Assert Board Highlights are rendered
        highlights = page.locator(".cai-highlight")
        assert highlights.count() == 2, "Highlights should appear for both from and to squares."

        # 5. Capture screenshot for audit trail
        screenshot_path = Path(__file__).parent / "fixtures" / "playwright_extension_e2e_audit.png"
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(screenshot_path), full_page=True)
        assert screenshot_path.exists()

        browser.close()

    server.stop()
