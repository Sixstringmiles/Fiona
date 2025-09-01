# Project: VRM Voice Avatar (Python)
#
# A modular, well-documented Python reference app that:
# 1) Listens to the user's microphone (offline STT via Vosk)
# 2) Sends recognized text to the OpenAI Chat Completions API
# 3) Speaks the AI response with low-latency streaming TTS (Piper)
# 4) Renders a VRM avatar (in a pywebview-embedded webpage using three.js + three-vrm)
# 5) Drives live facial/mouth animation in sync with speech while keeping STT non-blocking
#
# ──────────────────────────────────────────────────────────────────────────────
# Folder layout
# ├─ src/
# │  ├─ main.py                 # Orchestrator (async event loop)
# │  ├─ config.py               # Config/env and constants
# │  ├─ chat_client.py          # OpenAI Chat API wrapper
# │  ├─ stt_vosk.py             # Microphone -> text (Vosk)
# │  ├─ tts_piper.py            # Text -> streaming audio (Piper) + amplitude callbacks
# │  ├─ audio_viseme.py         # Amplitude → viseme/mouth curve helpers
# │  ├─ ui_webview.py           # pywebview window + Python<->JS bridge
# │  └─ utils.py                # Shared helpers (logging, threading, etc.)
# ├─ web/
# │  ├─ index.html              # three.js + three-vrm scene, VRM loader, bridge hooks
# │  └─ app.js                  # Avatar control code (mouth/eyes/blink/breath)
# ├─ requirements.txt
# └─ README.md
#
# Notes
# - This is a reference implementation using public, well-documented libs.
# - three-vrm handles VRM 0.x/1.0 expression/bones; we animate via JS bridge.
# - For best stability, bundle three.js + GLTFLoader + three-vrm locally later; CDN is used initially.
# - STT, Chat, TTS, and UI run concurrently; speech animation uses audio RMS/AGC-based viseme drive.
# - The code below is a ready-to-run baseline; see README for setup.


# ==============================
# File: requirements.txt
# ==============================
# Core
openai>=1.30.0
pywebview>=4.4.1
PySide6>=6.7.1

# Audio I/O
sounddevice>=0.4.6
numpy>=1.26.4

# STT
vosk>=0.3.45

# TTS
piper-tts>=1.3.0

# Emotion (open-source classifier)
transformers>=4.43.3
huggingface-hub>=0.24.0
# torch is required by transformers; install a CPU build if you don't have CUDA
torch>=2.2.0

# Utilities
requests>=2.32.3
tqdm>=4.66.4

# ==============================
# File: src/config.py
# ==============================
import os
from dataclasses import dataclass

@dataclass
class Config:
    # Backend selection: 'openai' or 'ollama'
    chat_backend: str = os.getenv("CHAT_BACKEND", "openai")

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Ollama (local LLM)
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1")
    # Optional JSON string for extra options: {"temperature":0.7,"num_predict":512,"num_ctx":4096}
    ollama_options_json: str = os.getenv("OLLAMA_OPTIONS_JSON", "")

    # Vosk STT (download a model and point to it)
    # e.g. https://alphacephei.com/vosk/models
    vosk_model_path: str = os.getenv("VOSK_MODEL_PATH", "./models/vosk-model-small-en-us-0.15")
    stt_sample_rate: int = int(os.getenv("STT_SAMPLE_RATE", "16000"))

    # Piper TTS (download an ONNX voice from rhasspy/piper-voices; set path here)
    # e.g. en_US-lessac-medium.onnx or similar
    piper_model_path: str = os.getenv("PIPER_MODEL_PATH", "./voices/en_US-lessac-medium.onnx")

    # Audio
    tts_block_ms: int = int(os.getenv("TTS_BLOCK_MS", "32"))  # size of chunks sent to speaker/UI

    # UI
    window_title: str = os.getenv("APP_TITLE", "VRM Voice Avatar")
    ui_width: int = int(os.getenv("UI_WIDTH", "1280"))
    ui_height: int = int(os.getenv("UI_HEIGHT", "720"))
    vrm_autoload: str = os.getenv("VRM_AUTOLOAD", "")  # optional path to a .vrm file to auto-load

    # Misc
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

CFG = Config()


# ==============================
# File: src/utils.py
# ==============================
import logging
import queue

from .config import CFG


def setup_logging():
    logging.basicConfig(
        level=getattr(logging, CFG.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


class RingBuffer:
    """Simple thread-safe ring-ish buffer for amplitude/viseme data."""
    def __init__(self, maxlen=256):
        self.q = queue.Queue(maxsize=maxlen)

    def push(self, item):
        try:
            self.q.put_nowait(item)
        except queue.Full:
            # Drop oldest by getting one item
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(item)
            except queue.Full:
                pass

    def try_pop(self):
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None


# ==============================
# File: src/chat_client.py
# ==============================
from typing import List, Dict
from openai import OpenAI
from .config import CFG
import logging

log = logging.getLogger(__name__)


class ChatClient:
    """Tiny wrapper around OpenAI Chat Completions."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.client = OpenAI(api_key=api_key or CFG.openai_api_key)
        self.model = model or CFG.openai_model

    def ask(self, messages: List[Dict[str, str]]) -> str:
        """Send a chat completion request and return the assistant's text."""
        log.info("Calling OpenAI model=%s", self.model)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
        )
        # Defensive: prefer the first choice's message content
        content = resp.choices[0].message.content if resp.choices else ""
        log.debug("OpenAI response len=%d", len(content))
        return content


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


# ==============================
# File: src/audio_viseme.py
# ==============================
import numpy as np


class VisemeMapper:
    """Map raw audio int16 chunks → smooth [0..1] mouth openness values.
    This uses RMS energy + simple attack/release smoothing suitable for VTuber A/I/U/E/O blendshapes.
    """

    def __init__(self, attack=0.08, release=0.15, gain=1.8, floor=0.02):
        self.attack = attack
        self.release = release
        self.gain = gain
        self.floor = floor
        self._env = 0.0

    def step(self, pcm_i16: np.ndarray) -> float:
        if pcm_i16.size == 0:
            return max(0.0, self._env * (1.0 - self.release))
        rms = np.sqrt(np.mean((pcm_i16.astype(np.float32) / 32768.0) ** 2))
        target = max(0.0, min(1.0, self.gain * max(0.0, rms - self.floor) / (1.0 - self.floor)))
        if target > self._env:
            # attack
            self._env = self._env + (target - self._env) * self.attack
        else:
            # release
            self._env = self._env + (target - self._env) * self.release
        return float(self._env)

    def to_vowels(self, openness: float) -> dict:
        """Optional: split openness into simplistic A/I/U/E/O weights.
        Very rough heuristic so the mouth shapes vary a bit.
        """
        a = openness
        e = max(0.0, (openness - 0.3) * 1.2)
        i = max(0.0, (0.7 - abs(openness - 0.7)) * 1.3)
        o = max(0.0, (openness - 0.5) * 0.8)
        u = max(0.0, (0.4 - openness) * 1.0)
        # Normalize to [0..1]
        def clamp(x): return max(0.0, min(1.0, x))
        return {"aa": clamp(a), "ee": clamp(e), "ih": clamp(i), "oh": clamp(o), "ou": clamp(u)}


# ==============================
# File: src/emotion.py
# ==============================
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import logging

log = logging.getLogger(__name__)

LABEL_MAP = {
    # Map GoEmotions-style labels to VRM expression names
    'joy': 'happy',
    'surprise': 'surprised',
    'sadness': 'sad',
    'anger': 'angry',
    'neutral': 'relaxed',
    'optimism': 'happy',
    'amusement': 'happy',
    'disappointment': 'sad',
    'annoyance': 'angry',
    'excitement': 'surprised',
    'fear': 'surprised',
    'love': 'happy',
}

@dataclass
class EmotionResult:
    label: str
    score: float

class EmotionEngine:
    \"\"\"Open-source emotion classifier with graceful fallback.

    Tries to load a lightweight GoEmotions distilled model from Hugging Face.
    If transformers/torch are unavailable, falls back to a tiny rule-based heuristic.
    \"\"\"
    def __init__(self, model_id: str = 'joeddav/distilbert-base-uncased-go-emotions-student'):
        self.model_id = model_id
        self._pipe = None
        try:
            from transformers import pipeline  # type: ignore
            self._pipe = pipeline('text-classification', model=model_id, return_all_scores=True, top_k=None)
            log.info('Loaded emotion model: %s', model_id)
        except Exception as e:
            log.warning('Emotion model unavailable (%s). Using heuristic.', e)
            self._pipe = None

    def analyze(self, text: str) -> EmotionResult:
        text = (text or '').strip()
        if not text:
            return EmotionResult('relaxed', 1.0)
        if self._pipe is None:
            # Heuristic keywords; very small and safe
            low = text.lower()
            if any(w in low for w in ['yay', 'great', 'glad', 'awesome', '😊', '😀', '😁']):
                return EmotionResult('happy', 0.7)
            if any(w in low for w in ['sorry', 'sad', 'unfortunately', '😢', '🙁']):
                return EmotionResult('sad', 0.7)
            if any(w in low for w in ['angry', 'mad', 'frustrat', '😠']):
                return EmotionResult('angry', 0.7)
            if any(w in low for w in ['wow', 'surpris', '!?', '😮']):
                return EmotionResult('surprised', 0.7)
            return EmotionResult('relaxed', 0.6)
        # Use model
        out: List[Dict[str, float]] = self._pipe(text)[0]  # list of {label, score}
        # Normalize labels and pick a mapped one
        best = max(out, key=lambda x: x['score'])
        base = best['label'].lower()
        mapped = LABEL_MAP.get(base, 'relaxed')
        score = float(best['score'])
        return EmotionResult(mapped, score)

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


# ==============================
# File: src/ui_webview.py
# ==============================
import json
import os
import threading
import time

import webview

from .config import CFG
import logging

log = logging.getLogger(__name__)


class Bridge:
    """Python methods callable from JavaScript: window.pywebview.api.*"""

    def __init__(self, app):
        self.app = app

    def start_listening(self):
        log.info("JS called start_listening")
        self.app.start_listening()
        return {"ok": True}

    def stop_listening(self):
        log.info("JS called stop_listening")
        self.app.stop_listening()
        return {"ok": True}

    def send_user_text(self, text: str):
        log.info("JS user text: %s", text)
        self.app.handle_user_text(text)
        return {"ok": True}


class AvatarUI:
    def __init__(self, app):
        self.app = app
        self.window: webview.Window | None = None

    def start(self):
        bridge = Bridge(self.app)
        self.window = webview.create_window(
            CFG.window_title,
            url=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web", "index.html")),
            js_api=bridge,
            width=CFG.ui_width,
            height=CFG.ui_height,
        )
        # Launch in a GUI thread; app continues
        t = threading.Thread(target=webview.start, kwargs={"debug": False}, daemon=True)
        t.start()
        # Give UI time to init
        time.sleep(1.0)
        # Pass autoload path if set
        if CFG.vrm_autoload:
            self.load_vrm(CFG.vrm_autoload)

    def eval_js(self, code: str):
        if self.window:
            try:
                self.window.evaluate_js(code)
            except Exception as e:
                log.warning("eval_js error: %s", e)

    def set_subtitle(self, text: str):
        js = json.dumps(text)
        self.eval_js(f"window.avatar && window.avatar.setSubtitle({js});")

    def set_mouth(self, openness: float, vowels: dict):
        # Push openness + 5-vowel payload to JS
        payload = json.dumps({"open": openness, **vowels})
        self.eval_js(f"window.avatar && window.avatar.updateMouth({payload});")

    def set_expression(self, name: str, weight: float, duration_ms: int = 200):
        self.eval_js(f"window.avatar && window.avatar.setExpression({json.dumps(name)}, {weight}, {duration_ms});")

    def set_emotion(self, name: str, weight: float, duration_ms: int = 300):
        """High-level VRM expression setter for basic emotions (happy/angry/sad/relaxed/surprised)."""
        self.eval_js(f"window.avatar && window.avatar.setEmotion({json.dumps(name)}, {weight}, {duration_ms});")

    def load_vrm(self, path: str):
        self.eval_js(f"window.avatar && window.avatar.loadVRM({json.dumps(path)});")

    def load_vrm(self, path: str):
        self.eval_js(f"window.avatar && window.avatar.loadVRM({json.dumps(path)});")


# ==============================
# File: src/main.py
# ==============================
import threading
import logging
import json
from typing import List, Dict

from .utils import setup_logging
from .config import CFG
from .chat_client import ChatClient as OpenAIChatClient
from .chat_ollama import ChatClient as OllamaChatClient
from .stt_vosk import VoskRecognizer
from .tts_piper import PiperTTS
from .ui_webview import AvatarUI
from .emotion import EmotionEngine


class App:
    def __init__(self):
        setup_logging()
        # Choose chat backend
        if CFG.chat_backend.lower() == 'ollama':
            self.chat = OllamaChatClient()
        else:
            self.chat = OpenAIChatClient()
        self.tts = PiperTTS()
        self.ui = AvatarUI(self)
        self.stt = VoskRecognizer(on_final=self.handle_user_text)
        self.emotion = EmotionEngine()
        self._listening = False

    # ── UI-initiated controls (called from JS via Bridge) ────────────────────
    def start_listening(self):
        if not self._listening:
            self.stt.start()
            self._listening = True
            self.ui.set_subtitle("🎤 Listening…")

    def stop_listening(self):
        if self._listening:
            self.stt.stop()
            self._listening = False
            self.ui.set_subtitle("")

    # ── Core flow ───────────────────────────────────────────────────────────
    def handle_user_text(self, text: str):
        text = (text or "").strip()
        if not text:
            return
        log = logging.getLogger(__name__)
        log.info("User said: %s", text)
        self.ui.set_subtitle(f"You: {text}")

        def _respond_pipeline():
            # Build chat messages
            messages: List[Dict[str, str]] = [
                {"role": "system", "content": "You are a friendly, concise assistant."},
                {"role": "user", "content": text},
            ]
            try:
                reply = self.chat.ask(messages)
            except Exception as e:
                log.exception("Chat backend error")
                reply = "Sorry, I ran into a local chatbot error."
            self.ui.set_subtitle(f"Assistant: {reply}")

            # Emotion from reply
            emo = self.emotion.analyze(reply)
            target_label = emo.label
            target_weight = min(1.0, max(0.3, emo.score))

            def _animate_emotion_fade(tts_thread: threading.Thread):
                self.ui.set_emotion(target_label, target_weight, 200)
                tts_thread.join()
                self.ui.set_emotion(target_label, 0.0, 600)
                self.ui.set_emotion('relaxed', 0.3, 400)
                self.ui.set_emotion('relaxed', 0.0, 600)

            # Speak and drive mouth animation in a background thread
            def _speak():
                self.tts.speak_stream(reply, on_viseme=lambda open_, vowels: self.ui.set_mouth(open_, vowels))
                self.ui.set_mouth(0.0, {"aa":0,"ee":0,"ih":0,"oh":0,"ou":0})

            tts_thread = threading.Thread(target=_speak, daemon=True)
            tts_thread.start()
            threading.Thread(target=_animate_emotion_fade, args=(tts_thread,), daemon=True).start()

        # Run the heavy pipeline outside the audio callback thread
        threading.Thread(target=_respond_pipeline, daemon=True).start()

    # ── Lifecycle ────────────────────────────────────────────────────────────
    def run(self):
        self.ui.start()
        # Optionally auto-start listening
        self.start_listening()
        # Keep main thread alive
        threading.Event().wait()


if __name__ == "__main__":
    App().run()


# ==============================
# File: web/index.html
# ==============================
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>VRM Voice Avatar</title>
    <style>
      html, body { margin: 0; height: 100%; background: #0a0a0a; color: #ddd; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica; }
      #toolbar { position: absolute; top: 10px; left: 10px; z-index: 10; display: flex; gap: 8px; align-items: center; }
      button { background: #1e1e1e; color: #e5e5e5; border: 1px solid #333; border-radius: 10px; padding: 8px 12px; cursor: pointer; }
      button:hover { background: #2a2a2a; }
      #subtitle { position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%); max-width: 80%; text-align: center; padding: 8px 12px; background: rgba(0,0,0,0.45); border-radius: 12px; }
      canvas { display: block; }
      input[type=file] { color: #ddd; }
      .badge { padding: 4px 8px; background:#222; border-radius: 8px; font-size: 12px; }
    </style>
    <!-- three.js & loaders (CDN for starter; consider bundling locally for production) -->
    <script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
    <script src="https://unpkg.com/three@0.160.0/examples/js/loaders/GLTFLoader.js"></script>
    <script src="https://unpkg.com/@pixiv/three-vrm@2.0.3/lib/three-vrm.min.js"></script>
  </head>
  <body>
    <div id="toolbar">
      <button onclick="window.pywebview?.api.start_listening()">Start 🎤</button>
      <button onclick="window.pywebview?.api.stop_listening()">Stop 📴</button>
      <label class="badge">Load VRM <input type="file" id="vrmfile" accept=".vrm" /></label>
      <input id="text" placeholder="Type and press Enter…" style="padding:8px;border-radius:10px;border:1px solid #333;background:#111;color:#eee;width:360px" />
    </div>
    <div id="subtitle"></div>
    <script src="app.js"></script>
  </body>
</html>


# ==============================
# File: web/app.js
# ==============================
(() => {
  const W = { scene:null, camera:null, renderer:null, controls:null, clock:null, light:null, vrm:null };
  const state = { mouth:0, vowels:{aa:0,ee:0,ih:0,oh:0,ou:0}, blink:0, t:0, emotions:{happy:0,angry:0,sad:0,relaxed:0,surprised:0} };

  const canvas = document.createElement('canvas');
  document.body.appendChild(canvas);

  function init() {
    W.renderer = new THREE.WebGLRenderer({ canvas, antialias:true, alpha:false });
    W.renderer.setSize(window.innerWidth, window.innerHeight);
    W.renderer.setPixelRatio(window.devicePixelRatio);

    W.scene = new THREE.Scene();
    W.scene.background = new THREE.Color(0x0a0a0a);

    W.camera = new THREE.PerspectiveCamera(35, window.innerWidth/window.innerHeight, 0.1, 100);
    W.camera.position.set(0, 1.4, 2.2);

    const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 1.0);
    hemi.position.set(0, 1, 0);
    W.scene.add(hemi);

    const dir = new THREE.DirectionalLight(0xffffff, 1.2);
    dir.position.set(1, 1.2, 1.5);
    W.scene.add(dir);

    const grid = new THREE.GridHelper(10, 10, 0x222222, 0x111111);
    grid.position.y = -1.0;
    W.scene.add(grid);

    W.clock = new THREE.Clock();
    window.addEventListener('resize', onResize);
    onResize();

    animate();
  }

  function onResize() {
    W.camera.aspect = window.innerWidth / window.innerHeight;
    W.camera.updateProjectionMatrix();
    W.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  function animate() {
    requestAnimationFrame(animate);
    const dt = W.clock.getDelta();
    state.t += dt;

    // Simple procedural blink
    const blinkSpeed = 6.0; // times/minute
    const phase = (state.t * blinkSpeed) % 1.0;
    let blink = 0.0;
    if (phase < 0.08) blink = 1.0 - (phase / 0.08); // close fast
    else if (phase < 0.16) blink = (phase - 0.08) / 0.08; // open fast
    state.blink = blink;

    // Idle breathing to move chest/head slightly
    if (W.vrm) {
      const s = 0.02 * Math.sin(state.t * 1.6);
      W.vrm.scene.position.y = s;
    }

    applyAvatarAnimation();
    W.renderer.render(W.scene, W.camera);
  }

  function applyAvatarAnimation() {
    if (!W.vrm) return;
    const em = W.vrm.expressionManager;
    if (em) {
      // Lip sync
      em.setValue('aa', state.vowels.aa ?? state.mouth);
      em.setValue('ee', state.vowels.ee ?? 0);
      em.setValue('ih', state.vowels.ih ?? 0);
      em.setValue('oh', state.vowels.oh ?? 0);
      em.setValue('ou', state.vowels.ou ?? 0);

      // Blink
      em.setValue('blinkL', state.blink);
      em.setValue('blinkR', state.blink);

      // Emotions (basic set)
      const E = state.emotions;
      em.setValue('happy', E.happy || 0);
      em.setValue('angry', E.angry || 0);
      em.setValue('sad', E.sad || 0);
      em.setValue('relaxed', E.relaxed || 0);
      em.setValue('surprised', E.surprised || 0);

      em.update();
    }
  }

  async function loadVRM(fileOrUrl) {
    const loader = new THREE.GLTFLoader();

    function onGLTF(gltf) {
      const vrm = THREE.VRM.from(gltf);
      if (W.vrm) {
        W.scene.remove(W.vrm.scene);
        W.vrm.dispose?.();
      }
      W.vrm = vrm;
      W.scene.add(vrm.scene);
      vrm.scene.rotation.y = Math.PI; // face camera
    }

    if (typeof fileOrUrl === 'string') {
      loader.load(fileOrUrl, (gltf) => {
        THREE.VRMUtils.removeUnnecessaryJoints(gltf.scene);
        THREE.VRMUtils.removeUnnecessaryVertices(gltf.scene);
        onGLTF(gltf);
      });
    } else {
      const file = fileOrUrl;
      const url = URL.createObjectURL(file);
      loader.load(url, (gltf) => {
        onGLTF(gltf);
        URL.revokeObjectURL(url);
      });
    }
  }

  const subtitleEl = document.getElementById('subtitle');
  const inputEl = document.getElementById('text');
  inputEl?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      const text = inputEl.value.trim();
      if (text) {
        window.pywebview?.api.send_user_text(text);
        inputEl.value = '';
      }
    }
  });

  const vrmInput = document.getElementById('vrmfile');
  vrmInput?.addEventListener('change', (e) => {
    const f = e.target.files?.[0];
    if (f) loadVRM(f);
  });

  function tweenEmotion(name, target, durationMs) {
    const start = state.emotions[name] || 0;
    const delta = target - start;
    const T = Math.max(1, durationMs || 200);
    const t0 = performance.now();
    function step(now){
      const u = Math.min(1, (now - t0)/T);
      state.emotions[name] = start + delta * (u<0.5 ? 2*u*u : -1+(4-2*u)*u); // easeInOutQuad
      if (u < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  // Expose a tiny JS API for Python to call
  window.avatar = {
    setSubtitle(text) { subtitleEl.textContent = text || ''; },
    updateMouth(payload) {
      state.mouth = payload.open ?? 0;
      state.vowels = payload;
    },
    setExpression(name, weight, durationMs) {
      tweenEmotion(name, weight, durationMs);
    },
    setEmotion(name, weight, durationMs) {
      // Alias for clarity
      tweenEmotion(name, weight, durationMs);
      // Optionally dampen others for exclusivity
      const others = ['happy','angry','sad','relaxed','surprised'].filter(n => n !== name);
      others.forEach(n => tweenEmotion(n, 0, durationMs));
    },
    loadVRM(path) { loadVRM(path); },
  };

  init();
})();


# ==============================
# File: src/chat_ollama.py
# ==============================
from __future__ import annotations
import json
import os
from typing import List, Dict
import requests
from .config import CFG
import logging

log = logging.getLogger(__name__)


class ChatClient:
    """Local LLM chat via Ollama's /api/chat endpoint.

    Requires `ollama serve` running locally and a pulled model (e.g. `ollama pull llama3.1`).
    """
    def __init__(self, host: str | None = None, model: str | None = None, options_json: str | None = None, timeout: float = 180.0):
        self.host = (host or CFG.ollama_host).rstrip('/')
        self.model = model or CFG.ollama_model
        self.timeout = timeout
        try:
            self.options = json.loads(options_json or CFG.ollama_options_json or '{}')
        except Exception:
            self.options = {}

    def ask(self, messages: List[Dict[str, str]]) -> str:
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {"temperature": 0.7, **self.options},
            "stream": False,
        }
        log.info("Ollama chat model=%s host=%s", self.model, self.host)
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        # Non-streaming response format: {'message': {'role': 'assistant','content': '...'}, ...}
        msg = data.get('message', {})
        content = msg.get('content')
        if not content and isinstance(data.get('messages'), list):
            # Some versions echo a list; fallback
            for m in data['messages'][::-1]:
                if m.get('role') == 'assistant' and m.get('content'):
                    content = m['content']; break
        return content or ""


# ==============================
# File: README.md
# ==============================
# VRM Voice Avatar (Python + OpenAI + Vosk + Piper)

A modular Python reference app that listens to your mic (offline STT via **Vosk**), chats with **OpenAI** (Chat Completions), speaks with **Piper** (local TTS), and animates a **VRM** avatar (via **three.js** + **three-vrm** inside **pywebview**). Concurrency ensures animation and STT remain smooth and non-blocking. Now with **emotion-driven expressions** and a **model/voice fetcher** to bundle open-source assets locally.

## Why these libraries?
- **Vosk** – offline STT, Apache-2.0 models for many languages.
- **Piper** – fast, local neural TTS (MIT); voices distributed on Hugging Face.
- **three-vrm** – VRM 0.x/1.0 expression support (happy/angry/sad/relaxed/surprised + lip shapes).
- **pywebview** – lightweight native window hosting our WebGL avatar.
- **Transformers** (optional) – open-source emotion classifier (GoEmotions distilled).

## Setup
1. **Install system audio deps**
   - macOS: `brew install portaudio`
   - Debian/Ubuntu: `sudo apt-get install libportaudio2`
2. **Create env and install Python deps**
   ```bash
   python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Fetch open-source models & voices (recommended)**
   ```bash
   python -m scripts.fetch_assets  # downloads into ./models and ./voices
   ```
   This fetches by default:
   - **Vosk**: `vosk-model-small-en-us-0.15` (Apache-2.0) → `./models/vosk-model-small-en-us-0.15`
   - **Piper**: `en_US-lessac-medium.onnx` (voice, CC-BY-4.0) → `./voices/en_US-lessac-medium.onnx`
   - **Emotion**: `joeddav/distilbert-base-uncased-go-emotions-student` → cached in `~/.cache/huggingface`

4. **Choose your chatbot backend**
   - **OpenAI (default):**
     ```bash
     export CHAT_BACKEND=openai
     export OPENAI_API_KEY=sk-...
     export OPENAI_MODEL=gpt-4o-mini
     ```
   - **Local (Ollama):**
     - Install and start Ollama (macOS: `brew install ollama` then `ollama serve`).
     - Pull a model: `ollama pull llama3.1` (or any other instruct-capable model).
     - Set env vars:
       ```bash
       export CHAT_BACKEND=ollama
       export OLLAMA_HOST=http://localhost:11434
       export OLLAMA_MODEL=llama3.1
       # Optional, tweak generation:
       export OLLAMA_OPTIONS_JSON='{"temperature":0.6,"num_predict":512,"num_ctx":4096}'
       ```

5. **Run**
   ```bash
   python -m src.main
   ```
   Load a `.vrm` in the UI and hit **Start** to talk.

## Emotion-driven expressions
- We run an open-source **emotion classifier** on each assistant reply to select one of: `happy`, `angry`, `sad`, `relaxed (neutral)`, `surprised`.
- The UI blends this expression over time while TTS plays, then fades back to relaxed.
- If `transformers`/`torch` aren’t present, a tiny keyword heuristic kicks in so the feature still works (just less smart).

## How lip sync works
- **Piper** emits audio chunks; we compute per-block RMS to estimate mouth openness.
- That openness is mapped into VRM expression weights (`aa`, `ih`, `ou`, `ee`, `oh`).
- This amplitude-based approach yields smooth, low-latency animation without phoneme alignment.

## Config (env vars)
- `VOSK_MODEL_PATH`, `PIPER_MODEL_PATH`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `VRM_AUTOLOAD`, etc. See `src/config.py`.

## Extending
- Swap STT to Whisper (local or API) by adding a `stt_whisper.py` with the same `on_final` callback signature.
- Add richer visemes with phoneme alignment if your TTS can emit phonemes.
- Use different emotions or map to custom VRM expressions in `web/app.js`.

## Troubleshooting
- **No audio output**: ensure PortAudio is installed; try another audio device index in `sounddevice`.
- **STT model not found**: check `VOSK_MODEL_PATH` or rerun the fetcher.
- **VRM not loading**: some VRM 0.x models may need updated three-vrm; try another model.
- **Torch wheels**: If `torch` fails to install, see https://pytorch.org/get-started/locally/ for a compatible wheel; the app still runs (with heuristic emotions) without `torch/transformers`.

## License
MIT for this template. Check the licenses of each dependency and any model/voice you download.

# ==============================
# File: scripts/fetch_assets.py
# ==============================
import os
import tarfile
from pathlib import Path
import requests
from tqdm import tqdm

# Optional: emotion model downloads are handled by transformers automatically.

VOSK_URL = 'https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip'
VOSK_ZIP = 'vosk-model-small-en-us-0.15.zip'
PIPER_HF_URL = 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx?download=true'
PIPER_JSON_URL = 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json?download=true'

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'models'
VOICES = ROOT / 'voices'


def _download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get('content-length', 0))
    with open(dest, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=dest.name) as pbar:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def fetch_vosk():
    zip_path = MODELS / VOSK_ZIP
    if not zip_path.exists():
        _download(VOSK_URL, zip_path)
    # Extract
    try:
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(MODELS)
        print('Vosk model extracted to', MODELS)
    except Exception as e:
        print('Failed to extract Vosk zip:', e)


def fetch_piper_voice():
    onnx_path = VOICES / 'en_US-lessac-medium.onnx'
    json_path = VOICES / 'en_US-lessac-medium.onnx.json'
    if not onnx_path.exists():
        _download(PIPER_HF_URL, onnx_path)
    if not json_path.exists():
        _download(PIPER_JSON_URL, json_path)
    print('Piper voice saved to', VOICES)


if __name__ == '__main__':
    fetch_vosk()
    fetch_piper_voice()
    print('Done.')
