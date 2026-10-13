"""Global hotkeys beyond the kill-switch."""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger("pc_agent.hotkeys")

VOICE_HOTKEY = "<ctrl>+<alt>+<space>"
_listener = None


def start_voice_hotkey(toggle: Callable[[], None]) -> None:
    global _listener
    if _listener is not None:
        return
    try:
        from pynput import keyboard
    except Exception as exc:
        logger.warning("Voice hotkey unavailable: %s", exc)
        return
    try:
        hotkey = keyboard.GlobalHotKeys({VOICE_HOTKEY: toggle})
        hotkey.daemon = True
        hotkey.start()
        _listener = hotkey
        logger.info("Voice hotkey armed: %s", VOICE_HOTKEY)
    except Exception as exc:
        logger.warning("Could not arm voice hotkey: %s", exc)


def stop_voice_hotkey() -> None:
    global _listener
    if _listener is not None:
        try:
            _listener.stop()
        except Exception:
            pass
        _listener = None
