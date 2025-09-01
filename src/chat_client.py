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