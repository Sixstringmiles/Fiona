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