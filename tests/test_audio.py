"""
Unit and Integration Tests for VoiceAssistant (Phase 5: Persian Voice Assistant & Audio).
"""

import time
import pytest
from pathlib import Path
import chess
from audio.voice_assistant import VoiceAssistant


@pytest.fixture
def voice_assistant(tmp_path: Path) -> VoiceAssistant:
    assistant = VoiceAssistant(
        voice=VoiceAssistant.DEFAULT_VOICE_MALE,
        cache_dir=tmp_path / "audio_cache",
        enable_audio=True,
        mock_playback=True,  # mock audio device playback for CI/tests
    )
    yield assistant
    assistant.close()


def test_01_init_and_voice_selection(tmp_path):
    assistant = VoiceAssistant(
        voice=VoiceAssistant.DEFAULT_VOICE_FEMALE,
        cache_dir=tmp_path / "test_cache",
        enable_audio=True,
        mock_playback=True,
    )
    assert assistant.voice == "fa-IR-DilaraNeural"
    assert assistant.cache_dir.exists()
    assistant.close()


def test_02_format_square_persian(voice_assistant):
    assert voice_assistant.format_square("e4") == "ای چهار"
    assert voice_assistant.format_square("d5") == "دی پنج"
    assert voice_assistant.format_square("f3") == "اف سه"
    assert voice_assistant.format_square("a1") == "آ یک"
    assert voice_assistant.format_square("h8") == "اچ هشت"
    assert voice_assistant.format_square("c6") == "سی شش"


def test_03_translate_moves_to_persian(voice_assistant):
    board = chess.Board()

    # 1. Normal Pawn move: e2 -> e4
    move_e4 = chess.Move.from_uci("e2e4")
    phrase_e4 = voice_assistant.translate_move_to_persian(move_e4, board)
    assert "پیاده" in phrase_e4
    assert "ای چهار" in phrase_e4

    # 2. Knight move: g1 -> f3
    board.push(move_e4)
    board.push_san("e5")
    move_nf3 = chess.Move.from_uci("g1f3")
    phrase_nf3 = voice_assistant.translate_move_to_persian(move_nf3, board)
    assert "اسب" in phrase_nf3
    assert "اف سه" in phrase_nf3

    # 3. Capture move: exd4
    board_cap = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    move_cap = chess.Move.from_uci("e4d5")
    phrase_cap = voice_assistant.translate_move_to_persian(move_cap, board_cap)
    assert "پیاده" in phrase_cap
    assert "دی پنج" in phrase_cap
    assert "زد" in phrase_cap

    # 4. Castling move: O-O
    board_castle = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    move_castle = chess.Move.from_uci("e1g1")
    phrase_castle = voice_assistant.translate_move_to_persian(move_castle, board_castle)
    assert "قلعه کوچک" in phrase_castle

    # 5. Checkmate move: Qxf7#
    board_mate = chess.Board("r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4")
    move_mate = chess.Move.from_uci("f3f7")
    phrase_mate = voice_assistant.translate_move_to_persian(move_mate, board_mate)
    assert "وزیر" in phrase_mate
    assert "اف هفت" in phrase_mate
    assert "کیش و مات!" in phrase_mate


def test_04_translate_evaluations(voice_assistant):
    # Equal position
    eq_eval = {"score_cp": 10, "win_chance_pct": 50.5}
    assert "متعادل" in voice_assistant.translate_evaluation_to_persian(eq_eval)

    # Decisive White advantage
    w_eval = {"score_cp": 320, "win_chance_pct": 85.0}
    assert "سفید برتری" in voice_assistant.translate_evaluation_to_persian(w_eval)

    # Black advantage
    b_eval = {"score_cp": -180, "win_chance_pct": 25.0}
    assert "سیاه برتری" in voice_assistant.translate_evaluation_to_persian(b_eval)

    # Mate in 1
    m_eval = {"is_mate": True, "mate_in": 1}
    assert "کیش و مات قطعی در 1 حرکت" in voice_assistant.translate_evaluation_to_persian(m_eval)


def test_05_synthesize_and_disk_cache(voice_assistant):
    test_text = "کیش و مات!"
    # Synthesize first time
    t0 = time.perf_counter()
    mp3_file = voice_assistant.synthesize_to_file(test_text)
    dt1 = (time.perf_counter() - t0) * 1000

    assert mp3_file is not None
    assert mp3_file.exists()
    assert mp3_file.stat().st_size > 1000  # valid audio data

    # Synthesize second time (cache hit)
    t1 = time.perf_counter()
    cached_file = voice_assistant.synthesize_to_file(test_text)
    dt2 = (time.perf_counter() - t1) * 1000

    assert cached_file == mp3_file
    # Cache hit should be sub-millisecond (< 5ms)
    assert dt2 < 10.0
    print(f"\n[CACHE TEST] Cold synthesis: {dt1:.1f}ms | Cached hit: {dt2:.3f}ms")


def test_06_non_blocking_queue_and_worker(voice_assistant):
    # Non-blocking check: speak() must return in < 5ms
    t0 = time.perf_counter()
    voice_assistant.speak("حرکت اول")
    voice_assistant.speak("حرکت دوم")
    call_dt = (time.perf_counter() - t0) * 1000
    assert call_dt < 10.0, "speak() must return immediately without blocking the caller!"

    # Priority speech clears previous
    voice_assistant.speak("هشدار فوری", priority=True)
    voice_assistant.stop_current_speech()
    assert voice_assistant._speech_queue.empty()


def test_07_muted_audio_flag(tmp_path):
    assistant = VoiceAssistant(
        cache_dir=tmp_path / "muted_cache",
        enable_audio=False,
        mock_playback=True,
    )
    assistant.speak("این جمله نباید پخش شود")
    assert assistant._speech_queue.empty()
    assistant.close()


def test_08_announce_ai_recommendation_and_threats(voice_assistant):
    analysis = {
        "best_move": "f3f7",
        "best_move_san": "Qxf7#",
        "is_mate": True,
        "mate_in": 1,
    }
    voice_assistant.announce_ai_recommendation(analysis)
    assert voice_assistant.last_spoken_text is not None
    assert "Qxf7#" in voice_assistant.last_spoken_text
    assert "مات" in voice_assistant.last_spoken_text

    voice_assistant.announce_threat("وزیر زیر ضرب است")
    assert "وزیر زیر ضرب است" in voice_assistant.last_spoken_text

    voice_assistant.announce_game_over("1-0")
    assert "سفید برنده شد" in voice_assistant.last_spoken_text
    voice_assistant.stop_current_speech()
