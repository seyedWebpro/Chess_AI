"""
test_stealth_audio.py: Comprehensive unit tests for stealth audio and earcon synthesizer (Phase 13).
Validates WAV generation, PCM sample headers, earcon sequence formatting, and mode switching.
"""

import pytest
import wave
from pathlib import Path
from audio.stealth_audio import EarconSynthesizer, StealthAudioController, StealthMode


@pytest.fixture
def earcon_synth(tmp_path):
    return EarconSynthesizer(cache_dir=tmp_path / "earcons")


def test_earcon_basic_sound_generation(earcon_synth):
    assert earcon_synth.tick_wav.exists()
    assert earcon_synth.chime_wav.exists()

    with wave.open(str(earcon_synth.tick_wav), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 44100
        assert w.getnframes() > 0


def test_earcon_square_synthesis(earcon_synth):
    # Test synthesizing "d4"
    # d = 4 ticks, 4 = 4 chimes
    wav_path = earcon_synth.synthesize_square_earcon("d4")
    assert wav_path.exists()

    with wave.open(str(wav_path), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 44100
        duration = w.getnframes() / float(w.getframerate())
        # Duration should be around 0.5 to 1.0 second
        assert 0.3 <= duration <= 1.5

    # Test synthesizing "h8" (maximum length)
    wav_h8 = earcon_synth.synthesize_square_earcon("h8")
    assert wav_h8.exists()
    with wave.open(str(wav_h8), "rb") as w:
        duration_h8 = w.getnframes() / float(w.getframerate())
        assert 0.5 <= duration_h8 <= 2.2


def test_stealth_controller_dispatch(earcon_synth, tmp_path):
    controller = StealthAudioController(mode=StealthMode.EARCON_CLICKS, mock_playback=True)
    controller.synthesizer = earcon_synth

    # Dispatches destination square audio without exceptions
    controller.notify_move("e2", "e4", "e4")
    assert (earcon_synth.cache_dir / "square_e4.wav").exists()

    # Switch to SILENT
    controller.set_mode(StealthMode.SILENT)
    controller.notify_move("g1", "f3", "Nf3")

    # Switch to WHISPER
    controller.set_mode(StealthMode.WHISPER)
    assert controller.mode == StealthMode.WHISPER
