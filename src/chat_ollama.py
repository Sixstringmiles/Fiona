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
