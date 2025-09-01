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