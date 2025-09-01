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