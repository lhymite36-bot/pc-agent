"""WebSocket voice hub: browser mic ↔ LiveVoiceSession ↔ browser playback."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Callable

from pc_agent.core.live_session import LiveVoiceSession

logger = logging.getLogger("pc_agent.voice_hub")


class VoiceHub:
    def __init__(
        self,
        api_key_fn: Callable[[], str | None],
        chat_fn: Callable[[str], str],
        voice_engine_fn: Callable,
        on_gemini_audio: Callable[[], None] | None = None,
    ) -> None:
        self.on_gemini_audio = on_gemini_audio
        self.api_key_fn = api_key_fn
        self.chat_fn = chat_fn
        self.voice_engine_fn = voice_engine_fn
        self._mode_on = False
        self._session: LiveVoiceSession | None = None
        self._clients: set = set()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_mode(self, on: bool) -> None:
        self._mode_on = on
        # Session lifecycle is driven by the browser WS (start/stop messages),
        # so the toggle just records intent; the page opens the WS and sends start.

    async def handle(self, ws) -> None:
        self._clients.add(ws)
        self._loop = asyncio.get_running_loop()
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if "text" in msg and msg["text"] is not None:
                    try:
                        data = json.loads(msg["text"])
                    except Exception:
                        continue
                    await self._on_text(ws, data)
                elif "bytes" in msg and msg["bytes"] is not None:
                    if self._session:
                        self._session.push_pcm(msg["bytes"])
        except Exception as exc:
            logger.debug("ws closed: %s", exc)
        finally:
            self._clients.discard(ws)
            if not self._clients:
                self._stop_session()

    async def _on_text(self, ws, data: dict) -> None:
        typ = data.get("type")
        if typ == "start":
            self._start_session(continuous=bool(data.get("continuous")))
            await ws.send_json({"type": "status", "state": "connecting"})
        elif typ == "stop":
            self._stop_session()
            await ws.send_json({"type": "status", "state": "idle"})
        elif typ == "ping":
            await ws.send_json({"type": "pong"})

    def _start_session(self, continuous: bool) -> None:
        with self._lock:
            if self._session and self._session.active:
                return
            key = self.api_key_fn()
            if not key:
                self._broadcast({"type": "error", "detail": "Add your Gemini API key in Settings first."})
                return
            loop = self._loop

            def on_pcm(pcm: bytes) -> None:
                if loop is None:
                    return
                asyncio.run_coroutine_threadsafe(self._send_bytes(pcm), loop)

            def on_event(ev: dict) -> None:
                if loop is None:
                    return
                asyncio.run_coroutine_threadsafe(self._send_json(ev), loop)

            if self.on_gemini_audio:
                try:
                    self.on_gemini_audio()  # silence any queued local voice
                except Exception:
                    pass
            self._session = LiveVoiceSession(
                api_key=key,
                on_pcm_out=on_pcm,
                on_event=on_event,
                chat_fn=self.chat_fn,
                voice_engine=self.voice_engine_fn(),
            )
            self._session.start(continuous=continuous)

    def _stop_session(self) -> None:
        with self._lock:
            if self._session:
                self._session.stop()
                self._session = None

    async def _send_bytes(self, data: bytes) -> None:
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send_bytes(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    async def _send_json(self, data: dict) -> None:
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    def _broadcast(self, data: dict) -> None:
        if self._loop:
            asyncio.run_coroutine_threadsafe(self._send_json(data), self._loop)
