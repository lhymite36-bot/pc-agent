"""Native mic/speaker I/O via sounddevice (PortAudio)."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable

logger = logging.getLogger("pc_agent.terminal_audio")

INPUT_RATE = 16000
OUTPUT_RATE = 24000
CHANNELS = 1
DTYPE = "int16"


def sounddevice_available() -> tuple[bool, str]:
    try:
        import sounddevice as sd  # noqa: F401
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def list_devices() -> list[dict]:
    import sounddevice as sd

    out = []
    for i, d in enumerate(sd.query_devices()):
        out.append(
            {
                "index": i,
                "name": d.get("name"),
                "max_input_channels": d.get("max_input_channels", 0),
                "max_output_channels": d.get("max_output_channels", 0),
                "default_samplerate": d.get("default_samplerate"),
            }
        )
    return out


def default_devices() -> dict:
    import sounddevice as sd

    return {"input": sd.default.device[0], "output": sd.default.device[1]}


class MicRecorder:
    """Push PCM16 mono @ 16 kHz chunks to a callback or internal queue."""

    def __init__(
        self,
        device: int | None = None,
        on_chunk: Callable[[bytes], None] | None = None,
        block_ms: int = 100,
    ) -> None:
        self.device = device
        self.on_chunk = on_chunk
        self.block = int(INPUT_RATE * block_ms / 1000)
        self._q: queue.Queue[bytes | None] = queue.Queue()
        self._stream = None
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> None:
        if self._active:
            return
        import sounddevice as sd

        def _cb(indata, frames, time_info, status):  # noqa: ARG001
            if status:
                logger.debug("mic status: %s", status)
            data = bytes(indata)
            if self.on_chunk:
                try:
                    self.on_chunk(data)
                except Exception:
                    logger.exception("mic on_chunk failed")
            else:
                self._q.put(data)

        self._stream = sd.RawInputStream(
            samplerate=INPUT_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=self.block,
            device=self.device,
            callback=_cb,
        )
        self._stream.start()
        self._active = True

    def stop(self) -> bytes:
        """Stop and return any leftover queued PCM (when not using on_chunk)."""
        if not self._active:
            return b""
        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
        except Exception:
            logger.exception("mic stop failed")
        self._stream = None
        self._active = False
        buf = bytearray()
        while True:
            try:
                chunk = self._q.get_nowait()
            except queue.Empty:
                break
            if chunk:
                buf.extend(chunk)
        return bytes(buf)

    def read_utterance(self, silence_sec: float = 1.2, max_sec: float = 30, min_sec: float = 0.6) -> bytes:
        """Record until silence after speech, or max_sec. Blocking."""
        import struct

        self.start()
        buf = bytearray()
        last_voice = time.time()
        started = time.time()
        heard = False
        try:
            while time.time() - started < max_sec:
                try:
                    chunk = self._q.get(timeout=0.2)
                except queue.Empty:
                    chunk = b""
                if chunk:
                    buf.extend(chunk)
                    n = len(chunk) // 2
                    if n:
                        samples = struct.unpack(f"<{n}h", chunk[: n * 2])
                        energy = sum(abs(s) for s in samples) / n
                        if energy > 400:
                            heard = True
                            last_voice = time.time()
                if heard and time.time() - last_voice >= silence_sec and len(buf) >= int(INPUT_RATE * 2 * min_sec):
                    break
        finally:
            self.stop()
        return bytes(buf)


class Speaker:
    """Low-latency PCM16 mono playback @ 24 kHz (Gemini Live/TTS output)."""

    def __init__(self, device: int | None = None, rate: int = OUTPUT_RATE) -> None:
        self.device = device
        self.rate = rate
        self._stream = None
        self._lock = threading.Lock()

    def write(self, pcm: bytes) -> None:
        if not pcm:
            return
        import sounddevice as sd

        with self._lock:
            if self._stream is None:
                self._stream = sd.RawOutputStream(
                    samplerate=self.rate,
                    channels=CHANNELS,
                    dtype=DTYPE,
                    device=self.device,
                    latency="low",
                )
                self._stream.start()
            self._stream.write(pcm)

    def close(self) -> None:
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
