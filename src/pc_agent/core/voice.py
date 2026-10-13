"""Realistic human voice: Gemini Live (default) → expressive TTS → offline SAPI fallback.

- Live native-audio: natural prosody, emotion, breaths, interruptions; streams PCM ASAP.
- TTS path: gemini-3.8-flash-tts with a conversational style prompt (not lite/flat).
- Offline: pyttsx3 / SAPI only as last resort, labeled robotic in settings.
"""

from __future__ import annotations

import asyncio
import io
import logging
import struct
import threading
import wave
from pathlib import Path
from typing import Callable, Iterator

from pc_agent.core.models import is_retryable
from pc_agent.config import (
    DEFAULT_VOICE,
    LIVE_MODEL,
    LIVE_MODEL_FALLBACKS,
    TTS_MODEL,
    TTS_MODEL_FALLBACKS,
    VOICE_CATALOG,
    VOICE_STYLE_PROMPT,
    data_dir,
    load_config,
    save_config,
)

logger = logging.getLogger("pc_agent.voice")

PREVIEW_LINE = (
    "Hey — I'm ready. Just say the word and I'll jump on it."
)


class VoiceEngine:
    def __init__(self, api_key: str | None) -> None:
        self.api_key = api_key
        self._client = None
        self._play_lock = threading.Lock()
        self._stop = threading.Event()
        if api_key:
            try:
                from google import genai

                from google.genai import types as _t
                self._client = genai.Client(
                    api_key=api_key,
                    http_options=_t.HttpOptions(
                        timeout=60_000,
                        retry_options=_t.HttpRetryOptions(attempts=1),
                    ),
                )
            except Exception as exc:
                logger.warning("genai client unavailable for voice: %s", exc)

    # --- settings helpers ---

    @staticmethod
    def catalog() -> list[dict]:
        return list(VOICE_CATALOG)

    @staticmethod
    def mode() -> str:
        return load_config().get("voice_mode", "live")

    @staticmethod
    def voice_id() -> str:
        return load_config().get("voice_id", DEFAULT_VOICE)

    @staticmethod
    def set_voice(voice_id: str) -> None:
        cfg = load_config()
        cfg["voice_id"] = voice_id
        save_config(cfg)

    @staticmethod
    def set_mode(mode: str) -> None:
        if mode not in {"live", "tts", "offline"}:
            raise ValueError(mode)
        cfg = load_config()
        cfg["voice_mode"] = mode
        save_config(cfg)

    # --- public speak ---

    def speak(self, text: str, on_chunk: Callable[[bytes], None] | None = None) -> dict:
        """Speak text using the best available path. Streams when possible."""
        if not text or not text.strip():
            return {"ok": False, "error": "empty text"}
        if not load_config().get("voice_enabled", True):
            return {"ok": True, "skipped": True, "reason": "voice_disabled"}

        self._stop.clear()
        mode = self.mode()
        styled = self._with_style(text)

        if mode == "live" and self._client is not None:
            result = self._speak_live(styled, on_chunk=on_chunk)
            if result.get("ok"):
                return result
            logger.info("Live voice failed (%s); trying TTS", result.get("error"))

        if mode in {"live", "tts"} and self._client is not None:
            result = self._speak_tts(styled, on_chunk=on_chunk)
            if result.get("ok"):
                return result
            logger.info("TTS failed (%s); trying offline", result.get("error"))

        return self._speak_offline(text)

    def stop(self) -> None:
        self._stop.set()

    def preview(self, voice_id: str | None = None) -> dict:
        """Generate a short preview clip for the voice picker."""
        vid = voice_id or self.voice_id()
        prev = self.voice_id()
        try:
            self.set_voice(vid)
            # Force TTS for previews (fast, deterministic clip)
            cfg = load_config()
            old_mode = cfg.get("voice_mode", "live")
            cfg["voice_mode"] = "tts"
            save_config(cfg)
            out = data_dir() / f"preview_{vid}.wav"
            result = self._speak_tts(self._with_style(PREVIEW_LINE), save_path=out)
            cfg["voice_mode"] = old_mode
            save_config(cfg)
            if result.get("ok"):
                result["path"] = str(out)
                result["voice_id"] = vid
            return result
        finally:
            self.set_voice(prev)

    # --- Live native audio ---

    def _speak_live(
        self,
        text: str,
        on_chunk: Callable[[bytes], None] | None = None,
    ) -> dict:
        try:
            return asyncio.run(self._speak_live_async(text, on_chunk=on_chunk))
        except Exception as exc:
            return {"ok": False, "error": str(exc), "path": "live"}

    async def _speak_live_async(
        self,
        text: str,
        on_chunk: Callable[[bytes], None] | None = None,
    ) -> dict:
        from google.genai import types

        voice = self.voice_id()
        models = self._ranked("live", LIVE_MODEL_FALLBACKS)
        last_err = None
        pcm_chunks: list[bytes] = []

        for model in models:
            if not self._health().usable(model):
                continue
            try:
                config = {
                    "response_modalities": ["AUDIO"],
                    "system_instruction": VOICE_STYLE_PROMPT,
                    "speech_config": {
                        "voice_config": {
                            "prebuilt_voice_config": {"voice_name": voice}
                        }
                    },
                    "output_audio_transcription": {},
                }
                async with self._client.aio.live.connect(
                    model=model, config=config
                ) as session:
                    await session.send_client_content(
                        turns={
                            "role": "user",
                            "parts": [
                                {
                                    "text": (
                                        "Say this aloud naturally to the user, "
                                        "nothing else:\n\n" + text
                                    )
                                }
                            ],
                        },
                        turn_complete=True,
                    )
                    transcript_parts: list[str] = []
                    async for response in session.receive():
                        if self._stop.is_set():
                            break
                        sc = getattr(response, "server_content", None)
                        if sc is None:
                            continue
                        if getattr(sc, "interrupted", False):
                            break
                        mt = getattr(sc, "model_turn", None)
                        if mt and mt.parts:
                            for part in mt.parts:
                                inline = getattr(part, "inline_data", None)
                                if inline and inline.data:
                                    chunk = inline.data
                                    if isinstance(chunk, str):
                                        import base64

                                        chunk = base64.b64decode(chunk)
                                    pcm_chunks.append(chunk)
                                    if on_chunk:
                                        on_chunk(chunk)
                                    else:
                                        self._play_pcm_chunk(chunk, rate=24000)
                        ot = getattr(sc, "output_transcription", None)
                        if ot and getattr(ot, "text", None):
                            transcript_parts.append(ot.text)
                        if getattr(sc, "turn_complete", False):
                            break
                if pcm_chunks:
                    path = data_dir() / "last_utterance.wav"
                    self._write_wav(path, b"".join(pcm_chunks), rate=24000)
                    return {
                        "ok": True,
                        "engine": "live",
                        "model": model,
                        "voice": voice,
                        "transcript": "".join(transcript_parts).strip(),
                        "path": str(path),
                    }
                last_err = "no audio returned"
            except Exception as exc:
                last_err = str(exc)
                self._mark_failure(model, exc)
                logger.debug("live model %s failed: %s", model, exc)
                continue
        return {"ok": False, "error": last_err or "live failed", "path": "live"}

    # --- Expressive TTS ---

    def _speak_tts(
        self,
        text: str,
        on_chunk: Callable[[bytes], None] | None = None,
        save_path: Path | None = None,
    ) -> dict:
        from google.genai import types

        voice = self.voice_id()
        models = self._ranked("tts", TTS_MODEL_FALLBACKS)
        last_err = None

        for model in models:
            if not self._health().usable(model):
                continue
            # Skip known-flat lite models explicitly
            if "lite" in model or "3.1-flash-tts" in model:
                continue
            try:
                prompt = f"{VOICE_STYLE_PROMPT}\n\nSay this:\n{text}"
                config = types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice
                            )
                        )
                    ),
                )
                # Stream if the SDK supports it for faster first audio
                pcm = bytearray()
                try:
                    stream = self._client.models.generate_content_stream(
                        model=model,
                        contents=prompt,
                        config=config,
                    )
                    for event in stream:
                        if self._stop.is_set():
                            break
                        cand = event.candidates[0] if event.candidates else None
                        if not cand or not cand.content:
                            continue
                        for part in cand.content.parts or []:
                            inline = getattr(part, "inline_data", None)
                            if inline and inline.data:
                                chunk = inline.data
                                if isinstance(chunk, str):
                                    import base64

                                    chunk = base64.b64decode(chunk)
                                pcm.extend(chunk)
                                if on_chunk:
                                    on_chunk(chunk)
                                else:
                                    self._play_pcm_chunk(chunk, rate=24000)
                except Exception:
                    # Non-streaming fallback
                    response = self._client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=config,
                    )
                    for part in response.candidates[0].content.parts:
                        inline = getattr(part, "inline_data", None)
                        if inline and inline.data:
                            chunk = inline.data
                            if isinstance(chunk, str):
                                import base64

                                chunk = base64.b64decode(chunk)
                            pcm.extend(chunk)

                if not pcm:
                    last_err = "empty TTS audio"
                    continue
                out = save_path or (data_dir() / "last_utterance.wav")
                self._write_wav(out, bytes(pcm), rate=24000)
                if on_chunk is None and not save_path:
                    # already played in stream; if non-stream, play now
                    pass
                elif save_path is None and on_chunk is None:
                    self._play_pcm_chunk(bytes(pcm), rate=24000)
                return {
                    "ok": True,
                    "engine": "tts",
                    "model": model,
                    "voice": voice,
                    "path": str(out),
                }
            except Exception as exc:
                last_err = str(exc)
                self._mark_failure(model, exc)
                logger.debug("TTS model %s failed: %s", model, exc)
        return {"ok": False, "error": last_err or "tts failed", "path": "tts"}

    # --- Offline robotic fallback ---

    def _speak_offline(self, text: str) -> dict:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            return {
                "ok": True,
                "engine": "offline",
                "label": "Robotic offline (SAPI / pyttsx3) — last-resort fallback",
                "voice": "system",
            }
        except Exception as exc:
            return {
                "ok": False,
                "error": f"offline TTS failed: {exc}",
                "engine": "offline",
                "label": "Robotic offline (SAPI / pyttsx3) — last-resort fallback",
            }

    # --- model ranking / failover ---

    def _ranked(self, kind: str, fallback: list[str]) -> list[str]:
        """Expressive TTS or Live models from models.list (cached), fallback list otherwise."""
        names: list[str] = []
        try:
            if self._client is not None:
                if not hasattr(self, "_catalog"):
                    from pc_agent.core.models import ModelCatalog
                    self._catalog = ModelCatalog(self._client, cache_path=data_dir() / "models_cache.json")
                names = self._catalog.names()
        except Exception:
            names = []
        out: list[str] = []
        for n in names:
            low = n.lower()
            if kind == "tts" and "tts" in low and "lite" not in low and "3.1-flash-tts" not in low:
                out.append(n)
            if kind == "live" and ("live" in low or "native-audio" in low):
                out.append(n)
        from pc_agent.core.models import _version_key
        out = sorted(set(out), key=_version_key, reverse=True)
        merged = list(dict.fromkeys(list(fallback) + out))
        return merged

    def _health(self):
        if not hasattr(self, "_health_obj"):
            from pc_agent.core.models import ModelHealth
            self._health_obj = ModelHealth(data_dir() / "model_health.json")
        return self._health_obj

    def _mark_failure(self, model: str, exc: Exception) -> None:
        from pc_agent.core.models import _code, retry_delay_sec
        code = _code(exc)
        if code == 404:
            self._health().mark_dead(model)
        elif code == 429:
            self._health().mark_cooling(model, retry_delay_sec(exc))
        elif code in (500, 503):
            self._health().mark_cooling(model, 30)

    # --- audio helpers ---

    @staticmethod
    def _with_style(text: str) -> str:
        # Keep the spoken content itself natural; style is in system/prompt.
        return text.strip()

    def _play_pcm_chunk(self, pcm: bytes, rate: int = 24000) -> None:
        """Best-effort low-latency playback. Uses sounddevice if present, else winsound/wav."""
        if self._stop.is_set() or not pcm:
            return
        with self._play_lock:
            try:
                import sounddevice as sd

                stream = getattr(self, "_out_stream", None)
                if stream is None or getattr(self, "_out_rate", None) != rate:
                    stream = sd.RawOutputStream(samplerate=rate, channels=1, dtype="int16", latency="low")
                    stream.start()
                    self._out_stream, self._out_rate = stream, rate
                # Continuous stream: chunks and sentences play back-to-back, no cut-offs
                stream.write(pcm)
                return
            except Exception:
                pass
            # Fallback: buffer to a temp wav and play (higher latency)
            try:
                path = data_dir() / "_chunk.wav"
                self._write_wav(path, pcm, rate=rate)
                if __import__("sys").platform == "win32":
                    import winsound

                    winsound.PlaySound(str(path), winsound.SND_FILENAME)
                else:
                    # Linux test env: skip audible play
                    pass
            except Exception as exc:
                logger.debug("play chunk failed: %s", exc)

    @staticmethod
    def _write_wav(path: Path, pcm: bytes, rate: int = 24000, channels: int = 1) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(pcm)
