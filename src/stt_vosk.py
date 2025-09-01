# ==============================
# File: src/stt_vosk.py
# ==============================
import json
import threading
import time
from typing import Callable, Optional

import sounddevice as sd
from vosk import Model, KaldiRecognizer

from .config import CFG
import logging

log = logging.getLogger(__name__)


class VoskRecognizer:
    """Streaming microphone STT using Vosk. Calls on_final(text) when a phrase finishes."""

    def __init__(self, on_final: Callable[[str], None], samplerate: int | None = None):
        self.on_final = on_final
        self.samplerate = samplerate or CFG.stt_sample_rate
        self.model = Model(CFG.vosk_model_path)
        self.rec = KaldiRecognizer(self.model, self.samplerate)
        self.stream: Optional[sd.InputStream] = None
        self.thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._last_speech_time = 0.0
        self._min_phrase_silence = 0.8  # seconds

    def _audio_cb(self, indata, frames, time_info, status):
        if status:
            log.warning("STT stream status: %s", status)
        if self._stop.is_set():
            raise sd.CallbackStop()
        if self.rec.AcceptWaveform(indata.tobytes()):
            res = json.loads(self.rec.Result())
            text = res.get("text", "").strip()
            if text:
                self._last_speech_time = time.time()
                # We treat this as a completed segment (Vosk chunked result)
                self.on_final(text)
        else:
            # Partial results update last speech time if non-empty
            pres = json.loads(self.rec.PartialResult())
            if pres.get("partial"):
                self._last_speech_time = time.time()

    def start(self):
        log.info("Starting Vosk microphone stream @ %d Hz", self.samplerate)
        self._stop.clear()
        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            dtype="int16",
            callback=self._audio_cb,
        )
        self.stream.start()
        # Monitor for long silences to flush final results if needed
        self.thread = threading.Thread(target=self._silence_watcher, daemon=True)
        self.thread.start()

    def _silence_watcher(self):
        while not self._stop.is_set():
            time.sleep(0.1)
            # If we've been silent for a while, force a final result flush
            if (time.time() - self._last_speech_time) > self._min_phrase_silence:
                res = json.loads(self.rec.FinalResult())
                text = res.get("text", "").strip()
                if text:
                    self.on_final(text)

    def stop(self):
        self._stop.set()
        if self.stream:
            try:
                self.stream.stop(); self.stream.close()
            except Exception:
                pass
        self.stream = None

