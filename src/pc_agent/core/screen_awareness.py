"""Optional periodic screen understanding. Off by default (privacy toggle)."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from pc_agent.config import load_config
from pc_agent.core import safety

logger = logging.getLogger("pc_agent.screen_awareness")


class ScreenAwareness:
    def __init__(
        self,
        brain,
        take_screenshot: Callable[[], dict],
        on_update: Callable[[str], None] | None = None,
    ) -> None:
        self.brain = brain
        self.take_screenshot = take_screenshot
        self.on_update = on_update
        self.last_summary: str = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="pc-agent-screen", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            cfg = load_config()
            interval = int(cfg.get("screen_awareness_interval_sec") or 45)
            if cfg.get("screen_awareness") and not safety.is_halted():
                try:
                    self.refresh()
                except Exception:
                    logger.exception("screen awareness tick failed")
            self._stop.wait(max(15, interval))

    def refresh(self) -> str:
        shot = self.take_screenshot()
        if not shot.get("ok"):
            return self.last_summary
        from pathlib import Path
        from google.genai import types

        path = shot.get("path")
        if not path or not Path(path).exists():
            return self.last_summary
        data = Path(path).read_bytes()
        prompt = (
            "Describe what's on screen in 2-4 short sentences. "
            "Call out errors, dialogs, or anything the user might ask about. "
            "Be factual."
        )
        try:
            response = self.brain.client.models.generate_content(
                model=self.brain.model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=prompt),
                            types.Part.from_bytes(data=data, mime_type="image/png"),
                        ],
                    )
                ],
                config=types.GenerateContentConfig(temperature=0.2),
            )
            self.last_summary = (response.text or "").strip()
            if self.on_update and self.last_summary:
                self.on_update(self.last_summary)
            return self.last_summary
        except Exception as exc:
            logger.debug("screen summarize failed: %s", exc)
            return self.last_summary
