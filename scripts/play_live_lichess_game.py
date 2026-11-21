"""
play_live_lichess_game.py: Autonomous Live Online Chess Player for Lichess in Google Chrome.
Grandmaster / Master Level Live Gauntlet (Stockfish Level 7 / 8).
Connects to Google Chrome Lichess tab via DOM scraper & AppleScript,
computes optimal moves with Stockfish 19 NNUE / Polyglot Hybrid Brain,
speaks the moves in Persian Neural TTS, and executes the moves
using biological WindMouse kinematics (Minimum Jerk Model) via NativeMacMouseController.
"""

import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime
import chess
import chess.pgn
from loguru import logger

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.hybrid_brain import HybridBrain
from core.maia_engine import Persona
from audio.voice_assistant import VoiceAssistant
from online.dom_parsers import pieces_dict_to_fen, infer_castling_rights
from automation.mouse_controller import HumanMouseController


SCRAPER_JS = Path(__file__).resolve().parent / "lichess_scraper.js"
APPLESCRIPT_RUNNER = Path(__file__).resolve().parent / "run_js_chrome.applescript"


def execute_chrome_js() -> dict:
    """Executes lichess_scraper.js inside Google Chrome Lichess tab via AppleScript."""
    js_code = SCRAPER_JS.read_text()
    cmd = ["osascript", str(APPLESCRIPT_RUNNER), js_code, "lichess.org"]
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


def play_autonomous_lichess_game(persona: Persona = Persona.GOD_MODE, max_moves: int = 50):
    """Plays the live Lichess game autonomously against high-level opponents."""
    print("\n" + "=" * 80)
    print("🤖   CHESS AI: AUTONOMOUS LIVE GRANDMASTER CHALLENGE (LICHESS)")
    print(f"    Target: Google Chrome [lichess.org - Stockfish Level 7/8]")
    print(f"    Brain Engine: Stockfish 19 NNUE (Persona: {persona.name})")
    print(f"    Kinematics: WindMouse Biological Minimum Jerk Automation")
    print(f"    Audio: Persian Neural Voice Assistant")
    print("=" * 80 + "\n")

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    live_pgn_path = log_dir / "live_lichess_level7_game.pgn"
    victory_png_path = log_dir / "lichess_level7_victory.png"

    activate_chrome()
    time.sleep(0.4)

    voice = VoiceAssistant(enable_audio=True)
    human_mouse = HumanMouseController(prefer_drag=False, fast_mode=False)

    game = chess.pgn.Game()
    game.headers["Event"] = "Grandmaster Challenge Match"
    game.headers["Site"] = "Lichess.org"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["Round"] = "1"
    game.headers["White"] = "Chess_AI (Stockfish 19 NNUE)"
    game.headers["Black"] = "Lichess Stockfish Level 7 (ELO ~2400)"

    pgn_node = game
    game_initialized = False
    last_board_after_our_move = None
    moves_played_count = 0
    consecutive_waits = 0

    with HybridBrain(persona=persona, use_opening_book=True) as brain:
        print("🟢 Connecting to Google Chrome Lichess game...")

        while moves_played_count < max_moves:
            # 1. Scrape live state from Chrome
            state = execute_chrome_js()
            if not state or state.get("error"):
                time.sleep(0.4)
                continue

            # 2. Check if game is over
            if state.get("isGameOver"):
                print("\n" + "=" * 80)
                print(f"🏁 GAME OVER DETECTED on Lichess! (Title: {state.get('title')})")
                print("=" * 80)
                voice.speak("بازی با موفقیت به پایان رسید.")
                break

            # 3. Check if it's our turn
            if not state.get("isOurTurn"):
                consecutive_waits += 1
                if consecutive_waits % 8 == 0:
                    print("⏳ Waiting for Stockfish Level 7's response on Lichess...", end="\r")
                time.sleep(0.3)
                continue

            consecutive_waits = 0
            orientation = state.get("orientation", "white")
            pieces = state.get("pieces", {})
            board_info = state.get("board", {})

            # 4. Construct Board from pieces
            active_color = "w" if orientation == "white" else "b"
            castling_rights = infer_castling_rights(pieces)
            fen = pieces_dict_to_fen(pieces, active_color=active_color, castling=castling_rights)
            board = chess.Board(fen)

            # Check if opponent has actually made their move since our last move
            if last_board_after_our_move is not None:
                if last_board_after_our_move.piece_map() == board.piece_map():
                    consecutive_waits += 1
                    if consecutive_waits % 8 == 0:
                        print("⏳ Waiting for Stockfish Level 7's response on Lichess...", end="\r")
                    time.sleep(0.3)
                    continue

            # Validate legal chess position (e.g. guard against OPPOSITE_CHECK while opponent is replying to a check)
            if not board.is_valid():
                time.sleep(0.3)
                continue

            # Initialize PGN with starting position if midgame
            if not game_initialized:
                if board.fen() != chess.STARTING_FEN:
                    game.setup(board)
                game_initialized = True

            # If opponent just moved, record opponent move into PGN
            if last_board_after_our_move is not None:
                for m_cand in last_board_after_our_move.legal_moves:
                    sim_b = last_board_after_our_move.copy()
                    sim_b.push(m_cand)
                    if sim_b.piece_map() == board.piece_map():
                        san_b = last_board_after_our_move.san(m_cand)
                        uci_b = m_cand.uci()
                        print(f"♟️ [Opponent] Black: {san_b:>6s} ({uci_b[:2]} → {uci_b[2:4]})")
                        pgn_node = pgn_node.add_variation(m_cand)
                        break

            if board.is_game_over():
                res_str = board.result()
                print("\n" + "=" * 80)
                print(f"🏁 CHECKMATE / GAME OVER! Result: {res_str}")
                print("=" * 80)
                voice.speak("کیش و مات قطعی! تبریک! بازی با برتری کامل به پایان رسید.")
                break

            # 5. Let HybridBrain decide the best move
            t0 = time.perf_counter()
            decision = brain.decide_move(board, time_limit=0.06)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            move = decision.get("move")
            if not move or move not in board.legal_moves:
                legal_list = list(board.legal_moves)
                if not legal_list:
                    print("No legal moves available. Game concluded.")
                    break
                move = legal_list[0]

            uci = move.uci()
            san = board.san(move)
            eval_str = decision.get("eval_str", "0.00")
            src = decision.get("source", "ENGINE")

            moves_played_count += 1
            print(f"\n🎯 [AI Move {moves_played_count}] White: {san:>6s} ({uci[:2]} → {uci[2:4]}) | Eval: {eval_str:>6s} | {src} ({dt_ms:.1f}ms)")
            pgn_node = pgn_node.add_variation(move)

            # Track board after our move
            after_b = board.copy()
            after_b.push(move)
            last_board_after_our_move = after_b

            # 6. Persian Voice Announcement
            try:
                phrase = voice.translate_move_to_persian(move, board)
                voice.speak(phrase)
            except Exception:
                pass

            # 7. Execute move using WindMouse hardware clicks
            from_sq = uci[:2]
            to_sq = uci[2:4]

            activate_chrome()
            time.sleep(0.05)

            bbox_tuple = (
                int(board_info["left"]),
                int(board_info["top"] + 88),  # Chrome macOS toolbar offset
                int(board_info["width"]),
                int(board_info["height"])
            )

            human_mouse.play_move(
                from_square=from_sq,
                to_square=to_sq,
                board_bbox=bbox_tuple,
                orientation_white_bottom=(orientation == "white"),
            )

            # Promotion safety: click destination again to select Queen in Lichess popup
            if len(uci) == 5:
                time.sleep(0.15)
                dest_x, dest_y = human_mouse.sampler.sample_square_from_bbox(
                    bbox_tuple, to_sq, (orientation == "white")
                )
                human_mouse.backend.click(dest_x, dest_y)

            time.sleep(0.7)

    voice.close()

    # Capture screenshot
    try:
        subprocess.run(["screencapture", "-x", str(victory_png_path)], capture_output=True)
        print(f"📸 Screenshot saved to: {victory_png_path}")
    except Exception:
        pass

    # Save PGN
    game.headers["Result"] = "*"
    try:
        with open(live_pgn_path, "w", encoding="utf-8") as f:
            exporter = chess.pgn.FileExporter(f)
            game.accept(exporter)
        print(f"\n📜 Match PGN saved to: {live_pgn_path}")
        print("\n--- FINAL PGN ---")
        print(str(game))
        print("-----------------\n")
    except Exception as e:
        logger.warning(f"PGN export note: {e}")

    print("✅ Autonomous Grandmaster challenge session completed successfully.")


if __name__ == "__main__":
    play_autonomous_lichess_game(persona=Persona.GOD_MODE, max_moves=30)
