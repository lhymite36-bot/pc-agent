"""System tray icon + menu."""

from __future__ import annotations

import logging
import threading
from typing import Callable

logger = logging.getLogger("pc_agent.tray")


def start_tray(
    open_ui: Callable[[], None],
    halt: Callable[[], None],
    resume: Callable[[], None],
    quit_app: Callable[[], None],
    toggle_voice: Callable[[], None] | None = None,
    voice_is_on: Callable[[], bool] | None = None,
) -> threading.Thread | None:
    try:
        import pystray
        from PIL import Image, ImageDraw
    except Exception as exc:
        logger.warning("tray unavailable: %s", exc)
        return None

    def _icon_image():
        img = Image.new("RGB", (64, 64), "#1a1f29")
        d = ImageDraw.Draw(img)
        d.ellipse((8, 8, 56, 56), fill="#6aa7ff")
        d.rectangle((28, 20, 36, 44), fill="#0f1115")
        return img

    def on_open(icon, item):  # noqa: ARG001
        open_ui()

    def on_halt(icon, item):  # noqa: ARG001
        halt()

    def on_resume(icon, item):  # noqa: ARG001
        resume()

    def on_voice(icon, item):  # noqa: ARG001
        if toggle_voice:
            toggle_voice()

    def on_quit(icon, item):  # noqa: ARG001
        quit_app()
        icon.stop()

    def voice_checked(item):  # noqa: ARG001
        return bool(voice_is_on and voice_is_on())

    items = [
        pystray.MenuItem("Open chat", on_open, default=True),
    ]
    if toggle_voice:
        items.append(pystray.MenuItem("Voice mode", on_voice, checked=voice_checked))
    items.extend([
        pystray.MenuItem("Halt (kill switch)", on_halt),
        pystray.MenuItem("Resume", on_resume),
        pystray.MenuItem("Quit", on_quit),
    ])
    icon = pystray.Icon("pc-agent", _icon_image(), "PC Agent", pystray.Menu(*items))
    t = threading.Thread(target=icon.run, name="pc-agent-tray", daemon=True)
    t.start()
    return t
