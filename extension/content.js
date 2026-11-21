/**
 * Chess AI Grandmaster Live Assistant - Extension Content Script (Manifest V3)
 * Sub-millisecond DOM scraping, WebSocket bridge, and In-Page Glassmorphic HUD overlay.
 */

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
    let hudElement = null;

    function log(...args) {
        if (CONFIG.debug) {
            console.log('%c[ChessAI Extension]', 'color: #38bdf8; font-weight: bold;', ...args);
        }
    }

    // Initialize in-page floating badge
    function initInPageHUD() {
        if (document.getElementById('chess-ai-inpage-hud')) return;

        hudElement = document.createElement('div');
        hudElement.id = 'chess-ai-inpage-hud';
        hudElement.style.cssText = `
            position: fixed;
            top: 18px;
            right: 22px;
            z-index: 999999;
            background: rgba(15, 23, 42, 0.92);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(56, 189, 248, 0.35);
            border-radius: 12px;
            padding: 10px 16px;
            color: #f8fafc;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.6);
            display: flex;
            flex-direction: column;
            gap: 6px;
            min-width: 180px;
            pointer-events: auto;
            transition: all 0.25s ease;
        `;

        hudElement.innerHTML = `
            <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px;">
                <span style="font-weight: bold; font-size: 13px; color: #38bdf8;">♟ CHESS AI</span>
                <span id="cai-status" style="font-size: 10px; color: #ef4444; font-weight: bold;">● OFFLINE</span>
            </div>
            <div style="display: flex; align-items: baseline; justify-content: space-between;">
                <span id="cai-move" style="font-size: 18px; font-weight: 800; color: #ffffff;">--</span>
                <span id="cai-eval" style="font-size: 13px; font-weight: 700; color: #10b981;">0.00</span>
            </div>
            <div id="cai-concept" style="font-size: 11px; color: #facc15; font-weight: 600; display: none;"></div>
        `;

        document.body.appendChild(hudElement);
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
                updateStatusBadge(true);
                triggerScan();
            };

            ws.onmessage = function (event) {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'ping') {
                        ws.send(JSON.stringify({ type: 'pong', timestamp: Date.now() }));
                    } else if (data.type === 'best_move') {
                        renderBestMove(data);
                    }
                } catch (e) {
                    log('Error parsing WS message:', e);
                }
            };

            ws.onclose = function () {
                isConnected = false;
                updateStatusBadge(false);
                setTimeout(initWebSocket, CONFIG.reconnectIntervalMs);
            };

            ws.onerror = function () {
                isConnected = false;
                updateStatusBadge(false);
            };
        } catch (e) {
            setTimeout(initWebSocket, CONFIG.reconnectIntervalMs);
        }
    }

    function updateStatusBadge(online) {
        const statusEl = document.getElementById('cai-status');
        if (statusEl) {
            statusEl.textContent = online ? '● LIVE' : '● OFFLINE';
            statusEl.style.color = online ? '#10b981' : '#ef4444';
        }
    }

    // Render best move on the HUD
    function renderBestMove(data) {
        initInPageHUD();
        const moveEl = document.getElementById('cai-move');
        const evalEl = document.getElementById('cai-eval');
        const conceptEl = document.getElementById('cai-concept');

        if (moveEl) moveEl.textContent = data.san || `${data.from} → ${data.to}`;
        if (evalEl) evalEl.textContent = data.eval || '0.00';

        if (conceptEl && data.concept_title) {
            conceptEl.textContent = `🎯 ${data.concept_title}`;
            conceptEl.style.display = 'block';
        }

        highlightBoardSquares(data.from, data.to);
    }

    function highlightBoardSquares(fromSq, toSq) {
        // Remove previous highlights
        document.querySelectorAll('.cai-highlight').forEach(el => el.remove());

        const board = findBoardElement();
        if (!board) return;

        [fromSq, toSq].forEach((sq, idx) => {
            if (!sq) return;
            const highlight = document.createElement('div');
            highlight.className = 'cai-highlight';
            highlight.style.cssText = `
                position: absolute;
                pointer-events: none;
                z-index: 100;
                background-color: ${idx === 0 ? 'rgba(56, 189, 248, 0.45)' : 'rgba(34, 197, 94, 0.45)'};
                border: 2px solid ${idx === 0 ? '#38bdf8' : '#22c55e'};
                box-sizing: border-box;
                border-radius: 4px;
            `;
            // Position according to square
            positionSquareOverlay(highlight, sq, board);
            board.appendChild(highlight);
        });
    }

    function positionSquareOverlay(el, sq, board) {
        const file = sq.charCodeAt(0) - 97; // a=0..h=7
        const rank = parseInt(sq[1], 10) - 1; // 1=0..8=7
        const isFlipped = board.classList.contains('flipped') || board.getAttribute('data-orientation') === 'black';

        const col = isFlipped ? 7 - file : file;
        const row = isFlipped ? rank : 7 - rank;

        el.style.width = '12.5%';
        el.style.height = '12.5%';
        el.style.left = `${col * 12.5}%`;
        el.style.top = `${row * 12.5}%`;
    }

    // Scrape Board
    function findBoardElement() {
        return document.querySelector('chess-board, wc-chess-board, div.board, cg-board, #chessboard');
    }

    function scrapeBoard() {
        const board = findBoardElement();
        if (!board) return null;

        const isFlipped = board.classList.contains('flipped') || board.getAttribute('data-orientation') === 'black';
        const orientation = isFlipped ? 'black' : 'white';
        const rect = board.getBoundingClientRect();

        const pieces = {};
        // Chess.com style
        const pieceEls = board.querySelectorAll('.piece');
        pieceEls.forEach(p => {
            const cls = Array.from(p.classList);
            let pieceCode = null;
            let square = null;

            cls.forEach(c => {
                if (/^[wb][prnbqk]$/.test(c)) {
                    pieceCode = c;
                }
                const sqMatch = c.match(/^square-(\d)(\d)$/);
                if (sqMatch) {
                    const col = parseInt(sqMatch[1], 10);
                    const row = parseInt(sqMatch[2], 10);
                    const file = String.fromCharCode(96 + col);
                    square = `${file}${row}`;
                }
            });

            if (pieceCode && square) {
                pieces[square] = pieceCode;
            }
        });

        // Lichess style (cg-board)
        if (Object.keys(pieces).length === 0) {
            const cgPieces = board.querySelectorAll('piece');
            cgPieces.forEach(p => {
                const cls = Array.from(p.classList);
                let color = cls.find(c => c === 'white' || c === 'black');
                let role = cls.find(c => ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king'].includes(c));
                let key = p.getAttribute('cg-key');
                if (color && role && key) {
                    const roleChar = role === 'knight' ? 'n' : role[0];
                    const colorChar = color === 'white' ? 'w' : 'b';
                    pieces[key] = `${colorChar}${roleChar}`;
                }
            });
        }

        return {
            platform: board.tagName.toLowerCase().includes('cg') ? 'lichess' : 'chess.com',
            orientation: orientation,
            pieces: pieces,
            bbox: {
                x: Math.round(rect.left),
                y: Math.round(rect.top),
                width: Math.round(rect.width),
                height: Math.round(rect.height)
            }
        };
    }

    function triggerScan() {
        if (!isConnected || !ws) return;
        const state = scrapeBoard();
        if (!state || Object.keys(state.pieces).length === 0) return;

        const stateStr = JSON.stringify(state.pieces) + state.orientation;
        if (stateStr === lastSentFen) return;
        lastSentFen = stateStr;

        ws.send(JSON.stringify({
            type: 'board_update',
            ...state,
            timestamp: Date.now()
        }));
    }

    // Set up DOM Observer
    function initObserver() {
        const board = findBoardElement();
        if (!board) {
            setTimeout(initObserver, 500);
            return;
        }

        const observer = new MutationObserver(() => {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(triggerScan, CONFIG.debounceMs);
        });

        observer.observe(board, { childList: true, subtree: true, attributes: true });
        log('DOM Observer active on chessboard element.');
    }

    // Bootstrap
    function start() {
        initInPageHUD();
        initWebSocket();
        initObserver();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', start);
    } else {
        start();
    }
})();
