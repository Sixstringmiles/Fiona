# ==============================
# File: src/tts_piper.py
# ==============================
import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice
from typing import Callable
from .config import CFG
from .audio_viseme import VisemeMapper
import logging

log = logging.getLogger(__name__)


class PiperTTS:
    """Stream TTS audio to speakers and drive a callback with viseme amplitudes.

    on_viseme: Callable[[float, dict], None] gets (openness, vowels_dict) per block.
    """

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or CFG.piper_model_path
        log.info("Loading Piper voice: %s", self.model_path)
        self.voice = PiperVoice.load(self.model_path)
        self.mapper = VisemeMapper()

    def speak_stream(self, text: str, on_viseme: Callable[[float, dict], None] | None = None):
        sr = self.voice.config.sample_rate
        block_ms = max(8, min(64, CFG.tts_block_ms))
        block_samples = int(sr * (block_ms / 1000.0))
        log.info("Speaking with Piper @ %d Hz, block=%d samples", sr, block_samples)

        with sd.OutputStream(samplerate=sr, channels=1, dtype='int16') as stream:
            buf = b''
            for raw_bytes in self.voice.synthesize_stream_raw(text):
                buf += raw_bytes
                while len(buf) >= block_samples * 2:  # int16 → 2 bytes
                    chunk = buf[: block_samples * 2]
                    buf = buf[block_samples * 2:]
                    pcm = np.frombuffer(chunk, dtype=np.int16)
                    stream.write(pcm)
                    if on_viseme:
                        openness = self.mapper.step(pcm)
                        on_viseme(openness, self.mapper.to_vowels(openness))