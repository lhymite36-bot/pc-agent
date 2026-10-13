"""Offline speech-to-text (default). Uses faster-whisper; downloads the model on first use."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

from pc_agent.config import data_dir

logger = logging.getLogger("pc_agent.local_stt")

DEFAULT_MODEL = "base"  # good CPU balance; "small" if user wants quality
COMPUTE = "int8"
_lock = threading.Lock()
_model = None
_model_name: str | None = None


def model_dir() -> Path:
    p = data_dir() / "whisper"
    p.mkdir(parents=True, exist_ok=True)
    return p


def available() -> tuple[bool, str]:
    try:
        import faster_whisper  # noqa: F401
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def _progress(msg: str, on_progress: Callable[[str], None] | None) -> None:
    logger.info(msg)
    if on_progress:
        try:
            on_progress(msg)
        except Exception:
            pass


def get_model(name: str = DEFAULT_MODEL, on_progress: Callable[[str], None] | None = None):
    global _model, _model_name
    with _lock:
        if _model is not None and _model_name == name:
            return _model
        from faster_whisper import WhisperModel

        _progress(f"Loading offline STT model '{name}' (first time downloads ~150MB)…", on_progress)
        _model = WhisperModel(
            name,
            device="cpu",
            compute_type=COMPUTE,
            download_root=str(model_dir()),
        )
        _model_name = name
        _progress(f"Offline STT ready ({name}, {COMPUTE}).", on_progress)
        return _model


def transcribe(
    pcm_16k_mono: bytes,
    language: str | None = "en",
    on_progress: Callable[[str], None] | None = None,
) -> str:
    """Transcribe raw PCM16 mono @ 16 kHz. Returns text or ''."""
    if not pcm_16k_mono or len(pcm_16k_mono) < 3200:
        return ""
    ok, detail = available()
    if not ok:
        raise RuntimeError(f"faster-whisper not installed: {detail}")

    import io
    import wave
    import numpy as np

    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm_16k_mono)
    bio.seek(0)

    # float32 mono for whisper
    with wave.open(bio, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    model = get_model(on_progress=on_progress)
    segments, _info = model.transcribe(
        audio,
        language=language,
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    parts = [s.text.strip() for s in segments if s.text and s.text.strip()]
    return " ".join(parts).strip()
