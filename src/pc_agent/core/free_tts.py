"""Free, fast TTS for CLI: edge-tts (neural, online-free) → SAPI/pyttsx3.

Target: start speaking in under ~1s. No Gemini quota.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import Callable

from pc_agent.config import data_dir

logger = logging.getLogger("pc_agent.free_tts")

# Warm, natural English voices (edge-tts)
DEFAULT_EDGE_VOICE = "en-US-JennyNeural"
EDGE_VOICES = [
    "en-US-JennyNeural",
    "en-US-GuyNeural",
    "en-GB-SoniaNeural",
    "en-IN-NeerjaNeural",
]


def speak_free(
    text: str,
    on_chunk: Callable[[bytes], None] | None = None,
    voice: str | None = None,
) -> dict:
    """Speak text with edge-tts if possible, else SAPI. Returns result dict."""
    text = (text or "").strip()
    if not text:
        return {"ok": False, "error": "empty"}
    # Prefer edge-tts (free Microsoft neural voices over the network, no Gemini)
    try:
        return _speak_edge(text, on_chunk=on_chunk, voice=voice or DEFAULT_EDGE_VOICE)
    except Exception as exc:
        logger.info("edge-tts unavailable (%s); SAPI", exc)
    return _speak_sapi(text)


def _speak_edge(text: str, on_chunk=None, voice: str = DEFAULT_EDGE_VOICE) -> dict:
    import edge_tts

    async def _run():
        communicate = edge_tts.Communicate(text, voice)
        out = data_dir() / "last_free_tts.mp3"
        # Stream to file then decode; edge-tts yields mp3 chunks
        with out.open("wb") as f:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])
                    if on_chunk:
                        # mp3 passthrough — player may not handle; collect then convert
                        pass
        return out

    out = asyncio.run(_run())
    # Play via sounddevice after decode with stdlib-only fallback
    played = _play_mp3(out, on_chunk=on_chunk)
    return {"ok": True, "engine": "edge-tts", "voice": voice, "path": str(out), "played": played}


def _play_mp3(path: Path, on_chunk=None) -> bool:
    """Best-effort play. Uses pygame if present, else ffplay/Windows."""
    try:
        import subprocess, sys
        if sys.platform == "win32":
            # PowerShell Media.SoundPlayer only does WAV — use start with default app briefly is bad.
            # Try winsound after convert via edge path: use pydub/ffmpeg if available.
            pass
        # Prefer sounddevice + audio decode via imageio_ffmpeg or simple MCI
        try:
            from pydub import AudioSegment
            import numpy as np
            import sounddevice as sd

            seg = AudioSegment.from_file(str(path))
            seg = seg.set_channels(1).set_frame_rate(24000)
            samples = np.array(seg.get_array_of_samples()).astype(np.float32)
            samples /= max(1, 1 << (8 * seg.sample_width - 1))
            pcm = (samples * 32767).astype(np.int16).tobytes()
            if on_chunk:
                # chunk ~100ms
                step = 24000 * 2 // 10
                for i in range(0, len(pcm), step):
                    on_chunk(pcm[i : i + step])
            else:
                sd.play(samples, 24000, blocking=True)
            return True
        except Exception:
            pass
        if sys.platform == "win32":
            # Use powershell with Windows.Media.Playback is heavy; fall back to SAPI instead
            return False
        subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)])
        return True
    except Exception as exc:
        logger.debug("play mp3 failed: %s", exc)
        return False


def _speak_sapi(text: str) -> dict:
    try:
        import pyttsx3

        eng = pyttsx3.init()
        # Snappier rate
        try:
            eng.setProperty("rate", 190)
        except Exception:
            pass
        eng.say(text)
        eng.runAndWait()
        return {
            "ok": True,
            "engine": "sapi",
            "label": "Windows SAPI (offline, free)",
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "engine": "sapi"}
