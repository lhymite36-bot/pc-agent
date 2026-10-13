"""Browser mic → Gemini Live (native audio) → browser playback.

PCM in: 16-bit LE mono 16 kHz. PCM out: 16-bit LE mono 24 kHz.
Falls back to STT (Gemini audio→text) + chat + TTS when Live is unavailable.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import threading
import time
from typing import Any, Callable

from pc_agent.config import LIVE_MODEL, LIVE_MODEL_FALLBACKS, VOICE_STYLE_PROMPT, data_dir, load_config
from pc_agent.core.models import ModelCatalog, ModelHealth, _code, retry_delay_sec

logger = logging.getLogger("pc_agent.live")
timing = logging.getLogger("pc_agent.timing")


class LiveVoiceSession:
    def __init__(
        self,
        api_key: str,
        on_pcm_out: Callable[[bytes], Any],
        on_event: Callable[[dict], Any],
        chat_fn: Callable[[str], str] | None = None,
        voice_engine=None,
    ) -> None:
        self.api_key = api_key
        self.on_pcm_out = on_pcm_out
        self.on_event = on_event
        self.chat_fn = chat_fn
        self.voice_engine = voice_engine
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pcm_in: asyncio.Queue | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._mode = "idle"  # idle|live|fallback
        self._health = ModelHealth(data_dir() / "model_health.json")

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, continuous: bool = True) -> None:
        if self.active:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(continuous,), name="pc-agent-live", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._loop and self._pcm_in is not None:
            try:
                self._loop.call_soon_threadsafe(self._pcm_in.put_nowait, None)
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        self._emit({"type": "status", "state": "idle", "engine": self._mode})

    def push_pcm(self, data: bytes) -> None:
        if not data or self._loop is None or self._pcm_in is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._pcm_in.put_nowait, data)
        except Exception:
            pass

    def _emit(self, event: dict) -> None:
        try:
            self.on_event(event)
        except Exception:
            logger.exception("on_event failed")

    def _run(self, continuous: bool) -> None:
        try:
            asyncio.run(self._async_main(continuous))
        except Exception:
            logger.exception("live session crashed")
            self._emit({"type": "status", "state": "error", "detail": "session crashed"})

    async def _async_main(self, continuous: bool) -> None:
        self._loop = asyncio.get_running_loop()
        self._pcm_in = asyncio.Queue(maxsize=64)
        self._emit({"type": "status", "state": "connecting"})
        ok = await self._try_live(continuous)
        if not ok and not self._stop.is_set():
            self._mode = "fallback"
            self._emit({"type": "status", "state": "listening", "engine": "fallback"})
            await self._fallback_loop()
        self._emit({"type": "status", "state": "idle", "engine": self._mode})

    def _models(self) -> list[str]:
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            cat = ModelCatalog(client, cache_path=data_dir() / "models_cache.json")
            names = cat.names()
        except Exception:
            names = []
        live = [n for n in names if "live" in n.lower() or "native-audio" in n.lower()]
        from pc_agent.core.models import _version_key
        live = sorted(set(live), key=_version_key, reverse=True)
        return list(dict.fromkeys(list(LIVE_MODEL_FALLBACKS) + live))

    async def _try_live(self, continuous: bool) -> bool:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        voice = load_config().get("voice_id") or "Sulafat"
        cfg = load_config()
        system = (
            f"You are {cfg.get('agent_name') or 'Atlas'}, a PC assistant. "
            f"{VOICE_STYLE_PROMPT} "
            f"Personality: {cfg.get('personality') or 'Witty, warm, and sharp.'} "
            "Keep spoken replies short."
        )
        for model in self._models():
            if self._stop.is_set():
                return False
            if not self._health.usable(model):
                continue
            try:
                config = {
                    "response_modalities": ["AUDIO"],
                    "system_instruction": system,
                    "speech_config": {
                        "voice_config": {"prebuilt_voice_config": {"voice_name": voice}}
                    },
                    "output_audio_transcription": {},
                    "input_audio_transcription": {},
                }
                t0 = time.time()
                async with client.aio.live.connect(model=model, config=config) as session:
                    self._mode = "live"
                    self._emit({"type": "status", "state": "listening", "engine": "live", "model": model})
                    timing.info("live_connected %.2fs model=%s", time.time() - t0, model)
                    send_task = asyncio.create_task(self._send_audio(session))
                    try:
                        async for response in session.receive():
                            if self._stop.is_set():
                                break
                            sc = getattr(response, "server_content", None)
                            if sc is None:
                                continue
                            if getattr(sc, "interrupted", False):
                                self._emit({"type": "status", "state": "interrupted"})
                                continue
                            mt = getattr(sc, "model_turn", None)
                            if mt and mt.parts:
                                for part in mt.parts:
                                    inline = getattr(part, "inline_data", None)
                                    if inline and inline.data:
                                        chunk = inline.data
                                        if isinstance(chunk, str):
                                            chunk = base64.b64decode(chunk)
                                        self.on_pcm_out(chunk)
                            it = getattr(sc, "input_transcription", None)
                            if it and getattr(it, "text", None):
                                self._emit({"type": "transcript", "role": "user", "text": it.text})
                            ot = getattr(sc, "output_transcription", None)
                            if ot and getattr(ot, "text", None):
                                self._emit({"type": "transcript", "role": "assistant", "text": ot.text})
                            if getattr(sc, "turn_complete", False) and not continuous:
                                break
                    finally:
                        send_task.cancel()
                        try:
                            await send_task
                        except Exception:
                            pass
                return True
            except Exception as exc:
                logger.warning("live model %s failed: %s", model, exc)
                code = _code(exc)
                if code == 404:
                    self._health.mark_dead(model)
                elif code == 429:
                    self._health.mark_cooling(model, retry_delay_sec(exc))
                elif code in (500, 503):
                    self._health.mark_cooling(model, 30)
                continue
        return False

    async def _send_audio(self, session) -> None:
        from google.genai import types

        assert self._pcm_in is not None
        while not self._stop.is_set():
            chunk = await self._pcm_in.get()
            if chunk is None:
                try:
                    await session.send_realtime_input(audio_stream_end=True)
                except Exception:
                    pass
                return
            try:
                await session.send_realtime_input(
                    audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
                )
            except Exception as exc:
                logger.debug("send audio failed: %s", exc)
                return

    async def _fallback_loop(self) -> None:
        """Collect ~1.5 s of silence-ended speech, STT via Gemini, chat, TTS back."""
        assert self._pcm_in is not None
        buf = bytearray()
        last_voice = time.time()
        silence_sec = 1.2
        min_bytes = 16000 * 2 * 1  # 1 s
        while not self._stop.is_set():
            try:
                chunk = await asyncio.wait_for(self._pcm_in.get(), timeout=0.3)
            except asyncio.TimeoutError:
                chunk = b""
            if chunk is None:
                break
            if chunk:
                buf.extend(chunk)
                # crude energy check
                import struct
                samples = struct.unpack(f"<{len(chunk)//2}h", chunk[: len(chunk) - (len(chunk) % 2)])
                if samples and (sum(abs(s) for s in samples) / len(samples)) > 400:
                    last_voice = time.time()
            if buf and time.time() - last_voice > silence_sec and len(buf) >= min_bytes:
                pcm = bytes(buf)
                buf.clear()
                await self._fallback_turn(pcm)

    async def _fallback_turn(self, pcm: bytes) -> None:
        self._emit({"type": "status", "state": "thinking", "engine": "fallback"})
        text = await asyncio.to_thread(self._stt, pcm)
        if not text:
            self._emit({"type": "status", "state": "listening", "engine": "fallback"})
            return
        self._emit({"type": "transcript", "role": "user", "text": text})
        reply = ""
        if self.chat_fn:
            reply = await asyncio.to_thread(self.chat_fn, text)
            self._emit({"type": "transcript", "role": "assistant", "text": reply})
        if reply and self.voice_engine:
            def _speak():
                def on_chunk(b: bytes) -> None:
                    self.on_pcm_out(b)
                # Force TTS path for browser playback (don't use offline SAPI)
                cfg = load_config()
                old = cfg.get("voice_mode")
                cfg["voice_mode"] = "tts"
                from pc_agent.config import save_config
                save_config(cfg)
                try:
                    self.voice_engine.speak(reply, on_chunk=on_chunk)
                finally:
                    cfg["voice_mode"] = old
                    save_config(cfg)
            await asyncio.to_thread(_speak)
        self._emit({"type": "status", "state": "listening", "engine": "fallback"})

    def _stt(self, pcm: bytes) -> str:
        try:
            from pc_agent.core.local_stt import available, transcribe
            ok, _ = available()
            if ok:
                text = transcribe(pcm)
                if text:
                    return text
        except Exception as exc:
            logger.warning("local STT failed: %s", exc)
        try:
            from google import genai
            from google.genai import types
            from pc_agent.config import FALLBACK_MODELS, data_dir, load_config
            from pc_agent.core.models import ModelCatalog, ModelHealth, ModelRouter
            from pc_agent.core.quota import format_quota_notice, is_quota_exhausted
            import io, wave

            client = genai.Client(api_key=self.api_key)
            health = ModelHealth(data_dir() / "model_health.json")
            cat = ModelCatalog(client, cache_path=data_dir() / "models_cache.json")
            ranked = cat.text_models(FALLBACK_MODELS, latency=health.latency_map())
            router = ModelRouter("stt", ranked or list(FALLBACK_MODELS),
                                 preferred=load_config().get("model"), health=health)
            bio = io.BytesIO()
            with wave.open(bio, "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
                wf.writeframes(pcm)
            wav = bio.getvalue()
            contents = [types.Content(role="user", parts=[
                types.Part.from_text(text="Transcribe this speech exactly. Reply with only the transcript."),
                types.Part.from_bytes(data=wav, mime_type="audio/wav"),
            ])]
            resp = router.call(lambda m: client.models.generate_content(model=m, contents=contents))
            return (resp.text or "").strip()
        except Exception as exc:
            logger.warning("STT failed: %s", exc)
            try:
                from pc_agent.core.quota import format_quota_notice, is_quota_exhausted
                if is_quota_exhausted(exc):
                    self._emit({"type": "status", "state": "quota", "detail": format_quota_notice(exc)})
            except Exception:
                pass
            return ""
