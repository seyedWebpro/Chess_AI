// ==UserScript==
// @name         Chess AI DOM Dual-Mode Bridge (Chess.com & Lichess)
// @namespace    http://tampermonkey.net/
// @version      1.0.0
// @description  Sub-millisecond DOM scraper and WebSocket bridge for Chess AI system.
// @author       Chess AI Architect
// @match        https://www.chess.com/*
// @match        https://lichess.org/*
// @grant        none
// @run-at       document-idle
// ==/UserScript==

(function () {
    'use strict';

    const CONFIG = {
        wsUrl: 'ws://127.0.0.1:8765',
        reconnectIntervalMs: 1500,
        debounceMs: 15,
        debug: true
    };

    let ws = null;
    let isConnected = false;
    let debounceTimer = null;
    let lastSentFen = null;

    function log(...args) {
        if (CONFIG.debug) {
            console.log('[ChessAI Bridge]', ...args);
        }
    }

    // Connect to Python WebSocket Server
    function initWebSocket() {
        if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
            return;
        }

        try {
            ws = new WebSocket(CONFIG.wsUrl);

            ws.onopen = function () {
                isConnected = true;
                log('Connected to Chess AI Python WebSocket Server.');
                // Trigger immediate initial board scan
                triggerScan();
            };

            ws.onmessage = function (event) {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'ping') {
                        ws.send(JSON.stringify({ type: 'pong', timestamp: Date.now() }));
                    } else if (data.type === 'best_move') {
                        renderBestMoveHint(data);
                    }
                } catch (e) { }
            };

            ws.onclose = function () {
                isConnected = false;
                log('Disconnected from server. Retrying in ' + CONFIG.reconnectIntervalMs + 'ms...');
                setTimeout(initWebSocket, CONFIG.reconnectIntervalMs);
            };

            ws.onerror = function (err) {
                isConnected = false;
            };
        } catch (e) {
            setTimeout(initWebSocket, CONFIG.reconnectIntervalMs);
        }
    }

    // Scrape Chess.com Board
    function scrapeChessCom() {
        const board = document.querySelector('chess-board, wc-chess-board, div.board');
        if (!board) return null;

        const pieceEls = board.querySelectorAll('.piece');
        const pieces = [];
        pieceEls.forEach(el => {
            pieces.push(el.className);
        });

        // Highlights
        const highlightEls = board.querySelectorAll('.highlight');
        const highlights = [];
        highlightEls.forEach(el => {
            highlights.push(el.className);
        });

        // Orientation
        const isFlipped = board.classList.contains('flipped');
        const orientation = isFlipped ? 'black' : 'white';

        return {
            site: 'chess.com',
            pieces: pieces,
            highlights: highlights,
            orientation: orientation,
            timestamp: Date.now() / 1000.0
        };
    }

    // Scrape Lichess Board
    function scrapeLichess() {
        const board = document.querySelector('cg-board');
        if (!board) return null;

        const pieceEls = board.querySelectorAll('piece');
        const pieces = [];
        pieceEls.forEach(el => {
            // Lichess piece classes: e.g. "white knight"
            const cls = el.className;
            // Key attribute if present or parent square
            const parentSq = el.closest('square');
            const sqName = parentSq ? parentSq.getAttribute('data-square') : null;
            pieces.push({
                class: cls,
                square: sqName,
                style: el.getAttribute('style') || ''
            });
        });

        // Orientation
        const container = document.querySelector('cg-container, .main-board');
        const orientation = (container && container.classList.contains('orientation-black')) ? 'black' : 'white';

        return {
            site: 'lichess',
            pieces: pieces,
            orientation: orientation,
            timestamp: Date.now() / 1000.0
        };
    }

    function triggerScan() {
        if (!isConnected) return;

        let data = null;
        if (window.location.hostname.includes('chess.com')) {
            data = scrapeChessCom();
        } else if (window.location.hostname.includes('lichess.org')) {
            data = scrapeLichess();
        }

        if (data && ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: 'dom_update',
                payload: data
            }));
        }
    }

    // Setup DOM MutationObserver
    function setupObserver() {
        const target = document.body;
        const observer = new MutationObserver(function () {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(triggerScan, CONFIG.debounceMs);
        });

        observer.observe(target, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['class', 'style']
        });

        log('DOM MutationObserver active.');
    }

    // Render Visual Best Move Hint & HUD Badge directly in the Browser
    function renderBestMoveHint(data) {
        log('Best move received from AI:', data.san, '(' + data.from + ' -> ' + data.to + ') | Eval:', data.eval);

        // Update or create in-browser HUD
        let hud = document.getElementById('chess-ai-hud');
        if (!hud) {
            hud = document.createElement('div');
            hud.id = 'chess-ai-hud';
            hud.style = 'position:fixed;top:18px;right:24px;z-index:999999;background:rgba(15,23,42,0.95);color:#38bdf8;padding:10px 18px;border-radius:12px;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;font-size:14px;box-shadow:0 10px 30px rgba(0,0,0,0.6);border:1px solid rgba(56,189,248,0.5);display:flex;gap:14px;align-items:center;pointer-events:none;backdrop-filter:blur(10px);transition:all 0.2s ease;';
            document.body.appendChild(hud);
        }
        hud.innerHTML = `
            <div style="display:flex;align-items:center;gap:6px;">
                <span style="font-size:18px;">♟️</span>
                <span style="color:#94a3b8;font-size:12px;text-transform:uppercase;letter-spacing:0.5px;">AI Move:</span>
                <strong style="color:#4ade80;font-size:16px;">${data.san || (data.from + ' → ' + data.to)}</strong>
            </div>
            <div style="width:1px;height:16px;background:rgba(255,255,255,0.15);"></div>
            <div style="display:flex;align-items:center;gap:6px;">
                <span style="color:#94a3b8;font-size:12px;text-transform:uppercase;letter-spacing:0.5px;">Eval:</span>
                <strong style="color:#facc15;font-size:15px;">${data.eval || '0.00'}</strong>
            </div>
            <div style="width:1px;height:16px;background:rgba(255,255,255,0.15);"></div>
            <div style="font-size:11px;color:#38bdf8;background:rgba(56,189,248,0.12);padding:2px 8px;border-radius:6px;border:1px solid rgba(56,189,248,0.3);">
                ${data.persona || 'Stockfish 19'}
            </div>
        `;

        applySquareHighlight(data.from, data.to);
    }

    function applySquareHighlight(fromSq, toSq) {
        document.querySelectorAll('.chess-ai-glow-from, .chess-ai-glow-to').forEach(el => {
            el.classList.remove('chess-ai-glow-from', 'chess-ai-glow-to');
        });

        if (!fromSq || !toSq) return;

        if (!document.getElementById('chess-ai-styles')) {
            const style = document.createElement('style');
            style.id = 'chess-ai-styles';
            style.innerHTML = `
                .chess-ai-glow-from {
                    box-shadow: inset 0 0 0 4px #22c55e, inset 0 0 20px rgba(34, 197, 94, 0.7) !important;
                    transition: box-shadow 0.15s ease !important;
                }
                .chess-ai-glow-to {
                    box-shadow: inset 0 0 0 4px #06b6d4, inset 0 0 20px rgba(6, 182, 212, 0.7) !important;
                    transition: box-shadow 0.15s ease !important;
                }
            `;
            document.head.appendChild(style);
        }

        // Lichess: square[data-square="e2"]
        const lichessFrom = document.querySelector('square[data-square="' + fromSq + '"]');
        const lichessTo = document.querySelector('square[data-square="' + toSq + '"]');
        if (lichessFrom) lichessFrom.classList.add('chess-ai-glow-from');
        if (lichessTo) lichessTo.classList.add('chess-ai-glow-to');

        // Chess.com: .square-XY
        const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
        const f1 = files.indexOf(fromSq[0]) + 1;
        const r1 = fromSq[1];
        const f2 = files.indexOf(toSq[0]) + 1;
        const r2 = toSq[1];

        const ccomFrom = document.querySelector('.square-' + f1 + r1);
        const ccomTo = document.querySelector('.square-' + f2 + r2);
        if (ccomFrom) ccomFrom.classList.add('chess-ai-glow-from');
        if (ccomTo) ccomTo.classList.add('chess-ai-glow-to');
    }

    // Initialize
    initWebSocket();
    setupObserver();
})();
