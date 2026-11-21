(() => {
    const b = document.querySelector(".round__app cg-board, main.round cg-board, .main-board cg-board, cg-board");
    if (!b) return JSON.stringify({ error: "no_board" });

    const rect = b.getBoundingClientRect();
    const sq = b.offsetWidth / 8;
    const files = ["a","b","c","d","e","f","g","h"];

    const container = document.querySelector("cg-container, .main-board");
    const isBlack = container && container.classList.contains("orientation-black");
    const orientation = isBlack ? "black" : "white";

    const pieces = {};
    b.querySelectorAll(":scope > piece, piece").forEach(p => {
        // Ignore ghost or animating dying pieces
        if (p.classList.contains("ghost") || p.classList.contains("fading")) return;

        const cls = p.className;
        const style = p.getAttribute("style") || "";
        const m = style.match(/translate\(([-0-9.]+)px,\s*([-0-9.]+)px\)/);
        if (!m) return;
        const px_x = parseFloat(m[1]);
        const px_y = parseFloat(m[2]);
        const fIdx = Math.round(px_x / sq);
        const rIdx = isBlack ? Math.round(px_y / sq) + 1 : 8 - Math.round(px_y / sq);
        if (fIdx < 0 || fIdx > 7 || rIdx < 1 || rIdx > 8) return;
        const fileChar = files[isBlack ? 7 - fIdx : fIdx];
        const sqName = fileChar + rIdx;

        const color = cls.includes("white") ? "w" : "b";
        let role = "p";
        if (cls.includes("knight")) role = "n";
        else if (cls.includes("bishop")) role = "b";
        else if (cls.includes("rook")) role = "r";
        else if (cls.includes("queen")) role = "q";
        else if (cls.includes("king")) role = "k";

        pieces[sqName] = color === "w" ? role.toUpperCase() : role;
    });

    const title = document.title || "";
    const isOurTurn = title.toLowerCase().includes("your turn");
    const isGameOver = document.querySelector(".result, .status, .game__meta__infos .result") !== null ||
                       document.body.innerText.includes("Checkmate") ||
                       document.body.innerText.includes("White is victorious") ||
                       document.body.innerText.includes("Black is victorious") ||
                       document.body.innerText.includes("Draw");

    return JSON.stringify({
        pieces: pieces,
        orientation: orientation,
        isOurTurn: isOurTurn,
        isGameOver: isGameOver,
        title: title,
        board: {
            left: rect.left,
            top: rect.top,
            width: rect.width,
            height: rect.height
        }
    });
})()
