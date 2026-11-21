(() => {
    // If a promotion modal is visible, automatically choose the Queen
    const promoQueen = document.querySelector(".promotion-piece.wq, .promotion-piece.bq, .promotion-menu .queen, [data-piece*='q']");
    if (promoQueen) {
        promoQueen.click();
    }

    const board = document.querySelector("chess-board, wc-chess-board, div.board");
    if (!board) return JSON.stringify({ error: "no_board" });

    const rect = board.getBoundingClientRect();
    const pieceEls = Array.from(board.querySelectorAll(".piece"));
    const pieces = pieceEls.map(p => p.className);

    const isFlipped = board.classList.contains("flipped");
    const orientation = isFlipped ? "black" : "white";

    const highlights = Array.from(board.querySelectorAll(".highlight")).map(h => h.className);

    const gameOverEl = document.querySelector(".game-over-modal, .modal-game-over-header-title, .game-over-dialog, .game-result, .board-dialog-component");
    const hasGameOverText = document.body.innerText.includes("won by checkmate") ||
                           document.body.innerText.includes("Checkmate") ||
                           document.body.innerText.includes("Game Over") ||
                           document.body.innerText.includes("Game over");

    const isGameOver = (gameOverEl !== null) || hasGameOverText;

    return JSON.stringify({
        pieces: pieces,
        orientation: orientation,
        highlights: highlights,
        isGameOver: isGameOver,
        board: {
            left: rect.left,
            top: rect.top,
            width: rect.width,
            height: rect.height
        }
    });
})()
