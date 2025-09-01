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