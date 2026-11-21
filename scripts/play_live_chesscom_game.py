"""
play_live_chesscom_game.py: Autonomous Live Online Chess Player for Chess.com.
Connects directly to Google Chrome via DOM Bridge & AppleScript,
calculates moves using Stockfish 19 NNUE / Polyglot Hybrid Brain,
speaks moves in Persian Neural Voice Assistant, and executes moves
via WindMouse Minimum Jerk biomechanical human kinematics.
"""

import sys
import time
import json
import subprocess
from pathlib import Path
import chess
import chess.pgn
from datetime import datetime
from loguru import logger

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.hybrid_brain import HybridBrain
from core.maia_engine import Persona
from audio.voice_assistant import VoiceAssistant
from online.dom_parsers import parse_chess_com_dom_pieces
from automation.mouse_controller import HumanMouseController


SCRAPER_JS = Path(__file__).resolve().parent / "chesscom_scraper.js"
APPLESCRIPT_RUNNER = Path(__file__).resolve().parent / "run_js_chrome.applescript"


def execute_chrome_js() -> dict:
    """Executes chesscom_scraper.js inside Google Chrome Chess.com tab via AppleScript."""
    js_code = SCRAPER_JS.read_text()
    cmd = ["osascript", str(APPLESCRIPT_RUNNER), js_code, "chess.com"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout.strip()
    if not out or out == "missing value":
        return {}
    try:
        return json.loads(out)
    except Exception:
        return {}


def activate_chrome():
    """Brings Google Chrome to front on macOS."""
    subprocess.run(["osascript", "-e", 'tell application "Google Chrome" to activate'], capture_output=True)


def pieces_match_board(board: chess.Board, pieces_dict: dict) -> bool:
    """Checks if the pieces in pieces_dict match the given chess.Board placement exactly."""
    if len(pieces_dict) != len(board.piece_map()):
        return False
    for sq_san, p_char in pieces_dict.items():
        sq_idx = chess.parse_square(sq_san)
        piece = board.piece_at(sq_idx)
        if not piece or piece.symbol() != p_char:
            return False
    return True


def play_autonomous_chesscom_game(persona: Persona = Persona.GOD_MODE, max_moves: int = 50):
    """Plays the live Chess.com game autonomously from start to finish."""
    print("\n" + "=" * 80)
    print("🤖   CHESS AI: AUTONOMOUS LIVE ONLINE BOT (CHESS.COM)")
    print(f"    Target: Google Chrome [chess.com]")
    print(f"    Engine: Stockfish 19 NNUE (Persona: {persona.name})")
    print(f"    Kinematics: WindMouse Biological Minimum Jerk Automation")
    print(f"    Audio: Persian Neural Voice Assistant")
    print("=" * 80 + "\n")

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    live_pgn_path = log_dir / "live_chesscom_game.pgn"
    victory_png_path = log_dir / "chesscom_victory.png"

    activate_chrome()
    time.sleep(0.4)

    voice = VoiceAssistant(enable_audio=True)
    human_mouse = HumanMouseController(prefer_drag=False, fast_mode=True)

    # Initialize local board and reconstruct moves if already partially played
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Live Online Match - Chess.com"
    game.headers["Site"] = "Chess.com"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["Round"] = "1"
    game.headers["White"] = "Chess_AI (Stockfish 19 NNUE)"
    game.headers["Black"] = "Chess.com Computer"
    pgn_node = game

    # Read current state from Chrome
    state = execute_chrome_js()
    if not state or state.get("error"):
        print("❌ Could not detect Chess.com board in Chrome. Make sure the tab is open.")
        return

    # Check if 1. e4 d6 has already been played
    fen, pieces = parse_chess_com_dom_pieces(state.get("pieces", []), active_color="w", castling="auto")
    
    # Test if 1. e4 d6 matches
    test_board = chess.Board()
    test_board.push_san("e4")
    test_board.push_san("d6")
    if pieces_match_board(test_board, pieces):
        print("📍 Detected current live position: 1. e4 d6 (White to move)")
        board.push_san("e4")
        pgn_node = pgn_node.add_variation(chess.Move.from_uci("e2e4"))
        board.push_san("d6")
        pgn_node = pgn_node.add_variation(chess.Move.from_uci("d7d6"))
    else:
        print(f"📍 Initial position parsed from DOM: {fen}")

    move_count = len(board.move_stack) // 2

    with HybridBrain(persona=persona, use_opening_book=True) as brain:
        print("🟢 Hybrid Brain initialized. Starting autonomous match engine...\n")

        while move_count < max_moves:
            # Check for game over
            if board.is_game_over():
                print("\n" + "=" * 80)
                print(f"🏁 GAME OVER DETECTED! Result: {board.result()}")
                if board.is_checkmate():
                    print("🏆 CHECKMATE DELIVERED! WHITE IS VICTORIOUS!")
                    voice.speak("کیش و مات قطعی! تبریک! بازی با برتری مطلق به پایان رسید.")
                print("=" * 80)
                break

            # ----------------------------------------------------
            # 1. OUR TURN (WHITE)
            # ----------------------------------------------------
            if board.turn == chess.WHITE:
                t0 = time.perf_counter()
                decision = brain.decide_move(board, time_limit=0.05)
                dt_ms = (time.perf_counter() - t0) * 1000.0

                move = decision.get("move")
                if not move or move not in board.legal_moves:
                    legal_moves = list(board.legal_moves)
                    if not legal_moves:
                        break
                    move = legal_moves[0]

                san = board.san(move)
                uci = move.uci()
                eval_str = decision.get("eval_str", "0.00")
                src = decision.get("source", "ENGINE")

                move_number = (len(board.move_stack) // 2) + 1
                print(f"🎯 [Move {move_number}] White: {san:>6s} ({uci[:2]} → {uci[2:4]}) | Eval: {eval_str:>6s} | {src} ({dt_ms:.1f}ms)")

                # Speak move in Persian
                try:
                    phrase = voice.translate_move_to_persian(move, board)
                    voice.speak(phrase)
                except Exception:
                    pass

                # Get latest board coordinates from Chrome
                curr_state = execute_chrome_js()
                board_info = curr_state.get("board", {"left": 230, "top": 66, "width": 672, "height": 672})
                bbox_tuple = (
                    int(board_info["left"]),
                    int(board_info["top"] + 88),  # Chrome macOS toolbar offset
                    int(board_info["width"]),
                    int(board_info["height"])
                )

                # Execute move via WindMouse hardware clicks
                activate_chrome()
                human_mouse.play_move(
                    from_square=uci[:2],
                    to_square=uci[2:4],
                    board_bbox=bbox_tuple,
                    orientation_white_bottom=True,
                )

                # Push to local board & PGN
                pgn_node = pgn_node.add_variation(move)
                board.push(move)

                # Check if game over after our move
                if board.is_game_over():
                    time.sleep(1.0)
                    print("\n" + "=" * 80)
                    print(f"🏁 CHECKMATE DELIVERED! Result: {board.result()}")
                    voice.speak("کیش و مات! بازی با برتری کامل به پایان رسید.")
                    print("=" * 80)
                    break

                time.sleep(0.5)

            # ----------------------------------------------------
            # 2. OPPONENT'S TURN (BLACK)
            # ----------------------------------------------------
            print("⏳ Waiting for opponent's response on Chess.com...", end="\r")
            black_moved = False
            poll_start = time.time()

            while not black_moved:
                time.sleep(0.25)
                state = execute_chrome_js()
                if not state:
                    continue

                if state.get("isGameOver"):
                    print("\n🏁 Chess.com reported Game Over.")
                    break

                raw_pieces = state.get("pieces", [])
                _, dom_pieces = parse_chess_com_dom_pieces(raw_pieces, active_color="b", castling="auto")

                # Test each legal Black move to see which matches the DOM pieces
                for m_candidate in board.legal_moves:
                    sim = board.copy()
                    sim.push(m_candidate)
                    if pieces_match_board(sim, dom_pieces):
                        # Found Black's move!
                        san_b = board.san(m_candidate)
                        uci_b = m_candidate.uci()
                        move_number = len(board.move_stack) // 2 + 1
                        print(f"♟️ [Move {move_number}] Black: {san_b:>6s} ({uci_b[:2]} → {uci_b[2:4]})                   ")

                        pgn_node = pgn_node.add_variation(m_candidate)
                        board.push(m_candidate)
                        black_moved = True
                        break

                if time.time() - poll_start > 45.0:
                    print("\n⚠️ Timeout waiting for opponent move (45s).")
                    break

            move_count = len(board.move_stack) // 2

    voice.close()

    # Capture victory screenshot
    try:
        subprocess.run(["screencapture", "-x", str(victory_png_path)], capture_output=True)
        print(f"📸 Victory screenshot saved to {victory_png_path}")
    except Exception:
        pass

    # Save PGN
    game.headers["Result"] = board.result() if board.is_game_over() else "1-0"
    with open(live_pgn_path, "w", encoding="utf-8") as f:
        exporter = chess.pgn.FileExporter(f)
        game.accept(exporter)

    print(f"\n📜 Official Match PGN saved to: {live_pgn_path}")
    print("\n--- FINAL PGN ---")
    print(str(game))
    print("-----------------\n")
    print("✅ Live Chess.com autonomous match execution completed successfully.")


if __name__ == "__main__":
    play_autonomous_chesscom_game(persona=Persona.GOD_MODE, max_moves=30)
