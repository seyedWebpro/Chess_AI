"""
stealth_audio.py: Discreet Stealth Audio Signaling & Acoustic Earcon Synthesizer.
Provides covert audio cues for chess moves without any on-screen visual clues:
- Whisper Voice (fast, quiet, low-frequency whisper).
- Earcon Clicks (innocuous acoustic mouse/keyboard clicks: N subtle ticks for file A-H, N chimes for rank 1-8).
- Binaural subtle pulses.
- Native multi-platform audio synthesis using pure Python (wave, struct, math, numpy).
"""

from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import math
import struct
import wave
import time
import threading
import queue
import subprocess
import sys
import os
import numpy as np
from loguru import logger


class StealthMode(Enum):
    NORMAL = "NORMAL"            # Clear full Persian voice
    WHISPER = "WHISPER"          # Quiet, low-pitch fast whisper
    EARCON_CLICKS = "EARCON_CLICKS"  # Subtle click/chime acoustic pulses resembling mouse clicks
    SILENT = "SILENT"            # Muted


class EarconSynthesizer:
    """
    Pure Python acoustic synthesizer generating subtle, non-intrusive sound cues.
    Outputs standard PCM 16-bit 44.1kHz mono WAV files.
    """

    SAMPLE_RATE = 44100

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or (Path(__file__).parent.parent / "cache" / "earcons")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._pre_generate_sounds()

    def _generate_click(self, freq: float = 1200.0, duration: float = 0.012, decay: float = 250.0) -> np.ndarray:
        """Generates a natural, gentle acoustic transient click (mouse-click sound)."""
        t = np.linspace(0, duration, int(self.SAMPLE_RATE * duration), False)
        envelope = np.exp(-decay * t)
        sine = np.sin(2 * np.pi * freq * t)
        noise = (np.random.rand(len(t)) - 0.5) * 0.15
        wave_data = (sine + noise) * envelope
        return (wave_data * 14000).astype(np.int16)

    def _generate_chime(self, freq: float = 650.0, duration: float = 0.035, decay: float = 90.0) -> np.ndarray:
        """Generates a soft, pleasant subtle tone chime for ranks."""
        t = np.linspace(0, duration, int(self.SAMPLE_RATE * duration), False)
        envelope = np.exp(-decay * t)
        sine = np.sin(2 * np.pi * freq * t)
        wave_data = sine * envelope
        return (wave_data * 12000).astype(np.int16)

    def _save_wav(self, file_path: Path, audio_samples: np.ndarray) -> Path:
        """Writes int16 samples to a standard WAV file."""
        with wave.open(str(file_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.SAMPLE_RATE)
            wav_file.writeframes(audio_samples.tobytes())
        return file_path

    def _pre_generate_sounds(self):
        """Generates and caches basic tick and chime WAV files."""
        self.tick_wav = self.cache_dir / "tick.wav"
        self.chime_wav = self.cache_dir / "chime.wav"

        if not self.tick_wav.exists():
            click_samples = self._generate_click(freq=1100.0, duration=0.012, decay=260.0)
            self._save_wav(self.tick_wav, click_samples)

        if not self.chime_wav.exists():
            chime_samples = self._generate_chime(freq=620.0, duration=0.035, decay=90.0)
            self._save_wav(self.chime_wav, chime_samples)

    def synthesize_square_earcon(self, square: str) -> Path:
        """
        Synthesizes an earcon sequence for a given square (e.g. 'd4'):
        - File 'd' (4th file) -> 4 rapid subtle ticks (spaced by 60ms)
        - 150ms pause
        - Rank '4' (4th rank) -> 4 gentle chimes (spaced by 75ms)
        """
        if len(square) != 2:
            square = "e4"
        file_char = square[0].lower()
        rank_char = square[1]

        file_idx = ord(file_char) - ord('a') + 1   # a=1 ... h=8
        rank_idx = int(rank_char) if rank_char.isdigit() else 4  # 1 ... 8
        file_idx = max(1, min(8, file_idx))
        rank_idx = max(1, min(8, rank_idx))

        target_file = self.cache_dir / f"square_{square.lower()}.wav"
        if target_file.exists():
            return target_file

        click = self._generate_click(freq=1200.0, duration=0.010, decay=280.0)
        chime = self._generate_chime(freq=640.0, duration=0.028, decay=110.0)

        gap_tick = np.zeros(int(self.SAMPLE_RATE * 0.055), dtype=np.int16)
        inter_pause = np.zeros(int(self.SAMPLE_RATE * 0.140), dtype=np.int16)
        gap_chime = np.zeros(int(self.SAMPLE_RATE * 0.070), dtype=np.int16)

        sequence: List[np.ndarray] = []

        # 1. Ticks for File (a=1..h=8)
        for i in range(file_idx):
            sequence.append(click)
            if i < file_idx - 1:
                sequence.append(gap_tick)

        # 2. Inter-sequence pause
        sequence.append(inter_pause)

        # 3. Chimes for Rank (1..8)
        for i in range(rank_idx):
            sequence.append(chime)
            if i < rank_idx - 1:
                sequence.append(gap_chime)

        full_audio = np.concatenate(sequence)
        self._save_wav(target_file, full_audio)
        return target_file


class StealthAudioController:
    """
    Manages discreet stealth audio playback:
    Coordinates between Whisper voice, Earcon acoustic ticks/chimes, and Normal speech.
    """

    def __init__(
        self,
        voice_assistant: Optional[Any] = None,
        mode: StealthMode = StealthMode.EARCON_CLICKS,
        mock_playback: bool = False,
    ):
        self.voice_assistant = voice_assistant
        self.mode = mode
        self.mock_playback = mock_playback
        self.synthesizer = EarconSynthesizer()
        self._current_process: Optional[subprocess.Popen] = None

    def set_mode(self, mode: StealthMode) -> None:
        """Switches stealth audio mode."""
        self.mode = mode
        logger.info(f"Stealth Audio mode set to: {mode.value}")

    def notify_move(self, from_sq: str, to_sq: str, san: Optional[str] = None) -> None:
        """
        Dispatches move audio based on active stealth mode.
        """
        if self.mode == StealthMode.SILENT:
            return

        if self.mode == StealthMode.EARCON_CLICKS:
            # Play acoustic earcon for destination square
            wav_path = self.synthesizer.synthesize_square_earcon(to_sq)
            self._play_file(wav_path)

        elif self.mode == StealthMode.WHISPER:
            # Short whispered text e.g. "e4" or "d دی چهار"
            whisper_text = f"{to_sq[0]} {to_sq[1]}"
            if self.voice_assistant:
                self.voice_assistant.speak(whisper_text)

        elif self.mode == StealthMode.NORMAL:
            if self.voice_assistant:
                # Normal voice assistant handles standard sentence
                pass

    def _play_file(self, file_path: Path) -> None:
        """Plays audio non-blockingly."""
        if self.mock_playback:
            logger.debug(f"[Mock Audio Play] {file_path.name}")
            return

        def _play():
            try:
                if sys.platform == "win32":
                    try:
                        import winsound
                        winsound.PlaySound(str(file_path), winsound.SND_FILENAME)
                    except Exception as win_err:
                        logger.debug(f"Windows audio play: {win_err}")
                elif sys.platform == "darwin" and os.path.exists("/usr/bin/afplay"):
                    proc = subprocess.Popen(
                        ["/usr/bin/afplay", str(file_path)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    proc.wait()
                elif sys.platform.startswith("linux") and os.path.exists("/usr/bin/aplay"):
                    proc = subprocess.Popen(
                        ["/usr/bin/aplay", "-q", str(file_path)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    proc.wait()
            except Exception as e:
                logger.error(f"Error playing earcon: {e}")

        t = threading.Thread(target=_play, daemon=True)
        t.start()
