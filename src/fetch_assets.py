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