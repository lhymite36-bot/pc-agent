"""Confirmation gates, kill-switch, and action logging."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pc_agent.config import CONFIRM_CATEGORIES, KILL_SWITCH_HOTKEY, load_config, log_path

logger = logging.getLogger("pc_agent.safety")

_halt = threading.Event()
_listener = None
_confirm_callback: Callable[[str, str], bool] | None = None


def is_halted() -> bool:
    return _halt.is_set()


def halt(reason: str = "kill-switch") -> None:
    _halt.set()
    log_action("system", "halt", {"reason": reason}, ok=True)
    logger.warning("Agent halted: %s", reason)


def resume() -> None:
    _halt.clear()
    log_action("system", "resume", {}, ok=True)


def set_confirm_callback(cb: Callable[[str, str], bool] | None) -> None:
    """Register UI callback: (category, description) -> approved?"""
    global _confirm_callback
    _confirm_callback = cb


def needs_confirmation(category: str) -> bool:
    cfg = load_config()
    if not cfg.get("confirm_dangerous", True):
        return False
    return category in CONFIRM_CATEGORIES


def confirm(category: str, description: str) -> bool:
    """Return True if the action may proceed."""
    if is_halted():
        return False
    if not needs_confirmation(category):
        return True
    approved = False
    cb = _confirm_callback
    if cb is not None:
        try:
            approved = bool(cb(category, description))
        except Exception as exc:
            logger.exception("Confirm callback failed")
            log_action(category, "confirm_error", {"error": str(exc)}, ok=False)
            return False
    else:
        # Native Windows MessageBox when no callback is registered.
        approved = _native_confirm(category, description)
        if approved is None:
            logger.warning("No confirm UI; denying %s: %s", category, description)
            log_action(category, "denied_no_ui", {"description": description}, ok=False)
            return False
    log_action(
        category,
        "confirm",
        {"description": description, "approved": approved},
        ok=approved,
    )
    return approved


def _native_confirm(category: str, description: str) -> bool | None:
    """Windows MessageBox; None if unavailable (Linux / headless)."""
    try:
        import ctypes
        import sys

        if sys.platform != "win32":
            return None
        MB_YESNO = 0x04
        MB_ICONWARNING = 0x30
        MB_TOPMOST = 0x40000
        result = ctypes.windll.user32.MessageBoxW(
            0,
            f"{description}\n\nCategory: {category}\n\nAllow this action?",
            "PC Agent — confirmation needed",
            MB_YESNO | MB_ICONWARNING | MB_TOPMOST,
        )
        return result == 6  # IDYES
    except Exception:
        return None


def log_action(category: str, action: str, details: dict, ok: bool = True) -> None:
    path = log_path()
    ts = datetime.now(timezone.utc).isoformat()
    line = f"{ts}\t{'OK' if ok else 'FAIL'}\t{category}\t{action}\t{details}\n"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
    except OSError as exc:
        logger.error("Could not write action log: %s", exc)


def start_kill_switch() -> None:
    """Listen for Ctrl+Alt+Esc globally. No-op if pynput unavailable."""
    global _listener
    if _listener is not None:
        return
    try:
        from pynput import keyboard
    except Exception as exc:
        logger.warning("Kill-switch unavailable: %s", exc)
        return

    def on_activate() -> None:
        halt("hotkey " + KILL_SWITCH_HOTKEY)

    try:
        hotkey = keyboard.GlobalHotKeys({KILL_SWITCH_HOTKEY: on_activate})
        hotkey.daemon = True
        hotkey.start()
        _listener = hotkey
        logger.info("Kill-switch armed: %s", KILL_SWITCH_HOTKEY)
    except Exception as exc:
        logger.warning("Could not arm kill-switch: %s", exc)


def stop_kill_switch() -> None:
    global _listener
    if _listener is not None:
        try:
            _listener.stop()
        except Exception:
            pass
        _listener = None
