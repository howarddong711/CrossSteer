import base64
import json
import os
from pathlib import Path
from typing import Optional

import requests

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from .paths import ASSET_DIR

if load_dotenv is not None:
    load_dotenv()

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/audio/speech"
SILICONFLOW_UPLOAD_URL = "https://api.siliconflow.cn/v1/uploads/audio/voice"
SILICONFLOW_MODEL = "IndexTeam/IndexTTS-2"

REFERENCE_AUDIO_PATH = os.path.join(ASSET_DIR, "reference.wav")
REFERENCE_TEXT = "This is a reference audio for text to speech synthesis"

_cached_voice_uri: Optional[str] = None

def upload_reference_voice(api_key: str) -> str:
    global _cached_voice_uri

    if _cached_voice_uri:
        return _cached_voice_uri

    if not os.path.exists(REFERENCE_AUDIO_PATH):
        raise FileNotFoundError(f"Reference audio not found: {REFERENCE_AUDIO_PATH}")

    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": SILICONFLOW_MODEL,
        "customName": "advwave-reference",
        "text": REFERENCE_TEXT,
    }

    with open(REFERENCE_AUDIO_PATH, "rb") as f:
        files = {"file": (os.path.basename(REFERENCE_AUDIO_PATH), f, "audio/wav")}
        response = requests.post(
            SILICONFLOW_UPLOAD_URL,
            headers=headers,
            data=data,
            files=files,
            timeout=120,
        )

    response.raise_for_status()
    uri = response.json().get("uri")
    if not uri:
        raise RuntimeError("Upload API response missing 'uri' field")

    _cached_voice_uri = uri
    print(f"Voice uploaded: {uri}")
    return uri

def prompt2audio(
    prompt: str,
    file_name: str,
    voice_id: int = 0,
    speed: float = 1.0,
    max_retries: Optional[int] = None,
    throttle_seconds: float = 0.0,
    allow_fallback: bool = True,
) -> str:
    if not SILICONFLOW_API_KEY:
        raise ValueError("SILICONFLOW_API_KEY not set")

    try:
        voice_uri = upload_reference_voice(SILICONFLOW_API_KEY)
    except Exception as exc:
        print(f"Reference voice upload failed: {exc}")
        if allow_fallback:
            print("Falling back to silent audio")
            generate_silent_audio(file_name)
            return file_name
        raise

    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": SILICONFLOW_MODEL,
        "input": prompt,
        "voice": voice_uri,
        "response_format": "wav",
        "stream": False,
        "sample_rate": 16000,
    }

    if 0.25 <= speed <= 4.0:
        payload["speed"] = speed

    max_retries = max_retries if max_retries is not None else 5
    response = None
    for attempt in range(max_retries):
        try:
            response = requests.post(
                SILICONFLOW_API_URL,
                headers=headers,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as exc:
            if attempt < max_retries - 1:
                import random
                import time

                backoff = (2 ** attempt) + random.random()
                print(f"TTS request failed (attempt {attempt + 1}/{max_retries}): {exc}")
                print(f"Retrying in {backoff:.1f}s...")
                time.sleep(backoff)
                continue
            if allow_fallback:
                print("TTS request failed, using silent audio fallback")
                generate_silent_audio(file_name)
                return file_name
            raise

    if response is None or response.status_code != 200:
        if allow_fallback:
            print("TTS request failed after retries, using silent audio fallback")
            generate_silent_audio(file_name)
            return file_name
        raise RuntimeError(f"TTS request failed with status {response.status_code if response else 'None'}")

    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        data = response.json()
        audio_b64 = data.get("audio") or data.get("data")
        if not audio_b64:
            raise RuntimeError(f"API returned JSON without audio payload: {json.dumps(data)[:200]}")
        audio_bytes = base64.b64decode(audio_b64)
    else:
        audio_bytes = response.content

    output_path = Path(file_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
        mp3_path = output_path.with_suffix(".mp3")
        mp3_path.write_bytes(audio_bytes)
        cmd = (
            "env -i PATH=/usr/bin:/bin:/usr/local/bin "
            f"ffmpeg -y -i {mp3_path} -ac 1 -ar 16000 {output_path}"
        )
        import subprocess

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            if allow_fallback:
                print(f"ffmpeg convert failed: {result.stderr[:200]}")
                print("Falling back to silent audio")
                generate_silent_audio(file_name)
                return file_name
            raise RuntimeError(f"ffmpeg convert failed: {result.stderr[:200]}")
        mp3_path.unlink(missing_ok=True)
        print(f"TTS saved: {file_name} (converted from mp3, size={output_path.stat().st_size} bytes)")
    else:
        output_path.write_bytes(audio_bytes)
        print(f"TTS saved: {file_name} (size={len(audio_bytes)} bytes)")

    if throttle_seconds and throttle_seconds > 0:
        import time

        time.sleep(float(throttle_seconds))

    return file_name

def generate_silent_audio(file_name: str, duration: float = 1.0, sample_rate: int = 16000) -> None:
    import numpy as np
    import struct

    output_path = Path(file_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_samples = int(duration * sample_rate)
    audio_data = np.zeros(num_samples, dtype=np.int16)

    with open(file_name, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(audio_data) * 2))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * 2))
        f.write(struct.pack("<H", 2))
        f.write(struct.pack("<H", 16))
        f.write(b"data")
        f.write(struct.pack("<I", len(audio_data) * 2))
        f.write(audio_data.tobytes())
