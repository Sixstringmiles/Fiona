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