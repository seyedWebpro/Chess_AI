"""
VoiceAssistant: Persian voice assistant for Chess AI.
Features:
- Microsoft Edge Neural TTS (Persian voices: fa-IR-FaridNeural & fa-IR-DilaraNeural).
- Natural Persian chess terminology translator (pieces, squares, captures, checks, castling).
- Non-blocking asynchronous audio playback queue (never stutters camera / CV thread).
- Disk caching for frequent phrases (0ms replay latency).
- Fallback to macOS native 'say' or silent log mode if offline.
"""

import os
import sys
import time
import hashlib
import asyncio
import subprocess
import threading
import queue
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import chess
from loguru import logger

try:
    import edge_tts
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False
    logger.warning("edge-tts is not installed. VoiceAssistant will use fallback synthesizers.")


class VoiceAssistant:
    """
    Manages Persian speech synthesis and audio announcements for chess moves,
    tactical threats, game results, and engine evaluations.
    """

    DEFAULT_VOICE_MALE = "fa-IR-FaridNeural"
    DEFAULT_VOICE_FEMALE = "fa-IR-DilaraNeural"

    # Persian chess phonetic dictionaries
    PIECE_NAMES_FA = {
        chess.PAWN: "پیاده",
        chess.KNIGHT: "اسب",
        chess.BISHOP: "فیل",
        chess.ROOK: "رخ",
        chess.QUEEN: "وزیر",
        chess.KING: "شاه",
    }

    FILE_NAMES_FA = {
        "a": "آ",
        "b": "بی",
        "c": "سی",
        "d": "دی",
        "e": "ای",
        "f": "اف",
        "g": "جی",
        "h": "اچ",
    }

    RANK_NAMES_FA = {
        "1": "یک",
        "2": "دو",
        "3": "سه",
        "4": "چهار",
        "5": "پنج",
        "6": "شش",
        "7": "هفت",
        "8": "هشت",
    }

    def __init__(
        self,
        voice: str = DEFAULT_VOICE_MALE,
        cache_dir: Optional[Union[str, Path]] = None,
        enable_audio: bool = True,
        mock_playback: bool = False,
    ):
        """
        Initialize the Persian Voice Assistant.
        
        Args:
            voice: Microsoft Edge Neural TTS voice ID.
            cache_dir: Directory to cache synthesized MP3 files.
            enable_audio: Master switch to enable/disable sound.
            mock_playback: If True, simulates playback without calling system audio (for tests).
        """
        self.voice = voice
        self.enable_audio = enable_audio
        self.mock_playback = mock_playback

        if cache_dir is None:
            self.cache_dir = Path(__file__).parent.parent / "cache" / "audio"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Thread-safe audio queue and worker thread
        self._speech_queue: queue.Queue = queue.Queue()
        self._is_running = True
        self._current_process: Optional[subprocess.Popen] = None
        self.last_spoken_text: Optional[str] = None
        self._worker_thread = threading.Thread(target=self._process_speech_queue, daemon=True)
        self._worker_thread.start()

        logger.info(f"Persian Voice Assistant initialized with voice: {self.voice}")

    def format_square(self, square_name: str) -> str:
        """Translates a square (e.g. 'e4') into Persian phonetics ('ای چهار')."""
        square_name = square_name.lower().strip()
        if len(square_name) != 2:
            return square_name
        file_part = self.FILE_NAMES_FA.get(square_name[0], square_name[0])
        rank_part = self.RANK_NAMES_FA.get(square_name[1], square_name[1])
        return f"{file_part} {rank_part}"

    def translate_move_to_persian(
        self,
        move: chess.Move,
        board_before: chess.Board,
        is_opponent: bool = False,
    ) -> str:
        """
        Translates a chess move into fluent, natural Persian speech.
        
        Examples:
        - e2e4: 'پیاده به خانه ای چهار'
        - Qxf7#: 'وزیر مهره اف هفت را زد، کیش و مات!'
        - O-O: 'قلعه کوچک'
        - g1f3: 'اسب به خانه اف سه'
        """
        prefix = "حرکت حریف: " if is_opponent else ""

        # 1. Castling
        if board_before.is_kingside_castling(move):
            desc = "قلعه کوچک"
            return f"{prefix}{desc}"
        elif board_before.is_queenside_castling(move):
            desc = "قلعه بزرگ"
            return f"{prefix}{desc}"

        moving_piece = board_before.piece_at(move.from_square)
        piece_type = moving_piece.piece_type if moving_piece else chess.PAWN
        piece_fa = self.PIECE_NAMES_FA.get(piece_type, "مهره")

        to_sq_name = chess.square_name(move.to_square)
        to_sq_fa = self.format_square(to_sq_name)

        is_capture = board_before.is_capture(move)

        if is_capture:
            captured_piece = board_before.piece_at(move.to_square)
            # En passant capture check
            if not captured_piece and board_before.is_en_passant(move):
                captured_name = "پیاده"
            elif captured_piece:
                captured_name = self.PIECE_NAMES_FA.get(captured_piece.piece_type, "مهره")
            else:
                captured_name = "مهره"
            move_phrase = f"{piece_fa}، {captured_name} خانه {to_sq_fa} را زد"
        else:
            move_phrase = f"{piece_fa} به خانه {to_sq_fa}"

        # Promotion
        if move.promotion:
            promo_fa = self.PIECE_NAMES_FA.get(move.promotion, "وزیر")
            move_phrase += f" و به {promo_fa} ارتقا یافت"

        # Check / Checkmate status after move is applied
        board_after = board_before.copy()
        board_after.push(move)

        if board_after.is_checkmate():
            move_phrase += "، کیش و مات!"
        elif board_after.is_check():
            move_phrase += "، کیش!"

        return f"{prefix}{move_phrase}"

    def translate_evaluation_to_persian(self, analysis: Dict[str, Any]) -> str:
        """
        Translates Stockfish numerical score into Persian natural commentary.
        """
        if analysis.get("is_mate"):
            mate_in = analysis.get("mate_in", 1)
            if mate_in > 0:
                return f"کیش و مات قطعی در {mate_in} حرکت برای سفید"
            else:
                return f"کیش و مات قطعی در {abs(mate_in)} حرکت برای سیاه"

        score_cp = analysis.get("score_cp", 0)
        win_chance = analysis.get("win_chance_pct", 50.0)

        if score_cp > 250:
            return "سفید برتری قاطعی دارد"
        elif score_cp > 90:
            return "سفید برتری مناسبی دارد"
        elif score_cp < -250:
            return "سیاه برتری کاملی دارد"
        elif score_cp < -90:
            return "سیاه برتری دارد"
        else:
            return "موقعیت متعادل و برابر است"

    def explain_concept(
        self,
        board: chess.Board,
        move: chess.Move,
        score_cp: Optional[float] = None,
        is_mate: bool = False,
        mate_in: Optional[int] = None,
        detailed: bool = True,
        speak_now: bool = True,
    ) -> str:
        """
        Generates conceptual Persian grandmaster commentary explaining the strategic
        motive behind the move, and optionally queues it for voice playback.
        """
        from core.concept_coach import ConceptCoach
        if not hasattr(self, "_concept_coach") or self._concept_coach is None:
            self._concept_coach = ConceptCoach()

        analysis = self._concept_coach.analyze_move(
            board=board,
            move=move,
            score_cp=score_cp,
            is_mate=is_mate,
            mate_in=mate_in
        )
        if detailed:
            commentary = analysis["explanation_fa"]
        else:
            base_phrase = self.translate_move_to_persian(move, board)
            commentary = f"{base_phrase}. {analysis['short_concept']}."

        if speak_now:
            self.speak(commentary)

        return commentary

    def speak(self, text: str, priority: bool = False):
        """
        Asynchronously adds Persian speech text to the audio queue.
        If priority=True, clears pending speeches to speak immediately.
        """
        if not self.enable_audio:
            logger.debug(f"[Muted Voice] {text}")
            return

        if priority:
            self.stop_current_speech()

        self.last_spoken_text = text
        self._speech_queue.put(text)
        logger.debug(f"[Voice Queue Added] {text}")

    def announce_move(
        self,
        move: chess.Move,
        board_before: chess.Board,
        is_opponent: bool = False,
    ):
        """Announces a chess move in Persian."""
        phrase = self.translate_move_to_persian(move, board_before, is_opponent=is_opponent)
        self.speak(phrase)

    def announce_ai_recommendation(self, analysis: Dict[str, Any]):
        """Announces the Stockfish best move suggestion in Persian."""
        best_san = analysis.get("best_move_san", "")
        best_uci = analysis.get("best_move", "")
        if not best_uci or len(best_uci) < 4:
            return

        from_sq = best_uci[:2]
        to_sq = best_uci[2:4]
        from_fa = self.format_square(from_sq)
        to_fa = self.format_square(to_sq)

        is_mate = analysis.get("is_mate", False)
        mate_text = " که منجر به مات می‌شود" if is_mate else ""

        phrase = f"حرکت پیشنهادی هوش مصنوعی: {best_san} از {from_fa} به {to_fa}{mate_text}."
        self.speak(phrase, priority=True)

    def announce_threat(self, threat_message: str):
        """Announces an urgent tactical alert in Persian."""
        phrase = f"هشدار تاکتیکی: {threat_message}"
        self.speak(phrase, priority=True)

    def announce_game_over(self, result_str: str):
        """Announces the final game result in Persian."""
        if "1-0" in result_str:
            msg = "بازی به پایان رسید. سفید برنده شد!"
        elif "0-1" in result_str:
            msg = "بازی به پایان رسید. سیاه پیروز شد!"
        elif "1/2" in result_str:
            msg = "بازی با نتیجه مساوی به پایان رسید."
        else:
            msg = f"بازی تمام شد. نتیجه: {result_str}"
        self.speak(msg, priority=True)

    def stop_current_speech(self):
        """Interrupts currently playing speech and clears the queue."""
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
                self._speech_queue.task_done()
            except queue.Empty:
                break

        if self._current_process and self._current_process.poll() is None:
            try:
                self._current_process.terminate()
            except Exception:
                pass
            self._current_process = None

    def synthesize_to_file(self, text: str) -> Optional[Path]:
        """
        Synthesizes Persian text to an MP3 file using Edge-TTS with persistent caching.
        Returns the path to the MP3 file, or None if failed.
        """
        # Create deterministic MD5 hash for cache key
        text_hash = hashlib.md5(f"{self.voice}:{text}".encode("utf-8")).hexdigest()
        cached_file = self.cache_dir / f"{text_hash}.mp3"

        if cached_file.exists() and cached_file.stat().st_size > 0:
            return cached_file

        if not HAS_EDGE_TTS:
            return None

        # Synthesize via edge_tts async communicate
        try:
            async def _synthesize():
                communicate = edge_tts.Communicate(text, self.voice)
                await communicate.save(str(cached_file))

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(lambda: asyncio.run(_synthesize())).result()
            else:
                asyncio.run(_synthesize())

            if cached_file.exists() and cached_file.stat().st_size > 0:
                return cached_file
        except Exception as e:
            logger.warning(f"Edge-TTS synthesis failed for '{text}': {e}. Falling back.")
            if cached_file.exists():
                cached_file.unlink(missing_ok=True)

        return None

    def _play_audio_file(self, file_path: Path):
        """Plays an audio file on macOS using /usr/bin/afplay."""
        if self.mock_playback:
            time.sleep(0.05)  # simulate brief playback
            return

        if sys.platform == "darwin" and os.path.exists("/usr/bin/afplay"):
            try:
                self._current_process = subprocess.Popen(
                    ["/usr/bin/afplay", str(file_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._current_process.wait()
            except Exception as e:
                logger.error(f"Error playing audio with afplay: {e}")
            finally:
                self._current_process = None
        else:
            logger.info(f"[Audio Playback] Simulating play for: {file_path}")

    def _fallback_macos_say(self, text: str):
        """Fallback to native macOS say command if edge-tts or internet is unavailable."""
        if self.mock_playback:
            return
        if sys.platform == "darwin" and os.path.exists("/usr/bin/say"):
            try:
                self._current_process = subprocess.Popen(
                    ["/usr/bin/say", text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._current_process.wait()
            except Exception as e:
                logger.warning(f"macOS say fallback failed: {e}")
            finally:
                self._current_process = None

    def _process_speech_queue(self):
        """Internal background consumer loop."""
        while self._is_running:
            try:
                text = self._speech_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                # 1. Primary: Edge Neural TTS (with MP3 caching)
                mp3_path = self.synthesize_to_file(text)
                if mp3_path and mp3_path.exists():
                    self._play_audio_file(mp3_path)
                else:
                    # 2. Secondary Fallback: macOS say
                    self._fallback_macos_say(text)
            except Exception as e:
                logger.error(f"Error in speech processing for '{text}': {e}")
            finally:
                self._speech_queue.task_done()

    def close(self):
        """Clean shutdown of background worker thread and queue."""
        self._is_running = False
        self.stop_current_speech()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        logger.info("Voice Assistant shutdown complete.")
