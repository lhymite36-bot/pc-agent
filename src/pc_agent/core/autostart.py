"""Start at login: Startup shortcut (Windows) or .desktop (Linux)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("pc_agent.autostart")


def enable_autostart() -> dict:
    if sys.platform == "win32":
        return _win_startup(True)
    return _linux_desktop(True)


def disable_autostart() -> dict:
    if sys.platform == "win32":
        return _win_startup(False)
    return _linux_desktop(False)


def _exe_command() -> str:
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}"'
    return f'"{sys.executable}" -m pc_agent'


def _win_startup(enable: bool) -> dict:
    try:
        startup = (
            Path(os.environ.get("APPDATA", ""))
            / "Microsoft"
            / "Windows"
            / "Start Menu"
            / "Programs"
            / "Startup"
        )
        startup.mkdir(parents=True, exist_ok=True)
        link = startup / "PC Agent.bat"
        if enable:
            link.write_text(f"@echo off\nstart \"\" {_exe_command()}\n", encoding="utf-8")
            return {"ok": True, "path": str(link)}
        if link.exists():
            link.unlink()
        return {"ok": True, "removed": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _linux_desktop(enable: bool) -> dict:
    try:
        autostart = Path.home() / ".config" / "autostart"
        autostart.mkdir(parents=True, exist_ok=True)
        desk = autostart / "pc-agent.desktop"
        if enable:
            desk.write_text(
                "\n".join(
                    [
                        "[Desktop Entry]",
                        "Type=Application",
                        "Name=PC Agent",
                        f"Exec={_exe_command()}",
                        "X-GNOME-Autostart-enabled=true",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            return {"ok": True, "path": str(desk)}
        if desk.exists():
            desk.unlink()
        return {"ok": True, "removed": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
