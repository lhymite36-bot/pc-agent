"""$0 offline command router — common intents with no model / no API quota."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Callable

# (patterns, handler_name, description)
# Patterns are matched case-insensitively against the full user line.


def _home() -> Path:
    return Path.home()


def _open_path(path: str | Path) -> str:
    path = str(Path(path).expanduser())
    if sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])
    return f"Opened {path}"


def _run(argv: list[str], shell: bool = False) -> str:
    subprocess.Popen(argv, shell=shell)
    return f"Launched {' '.join(argv) if isinstance(argv, list) else argv}"


HANDLERS: dict[str, Callable[[re.Match], str]] = {}


def _handler(name: str):
    def deco(fn):
        HANDLERS[name] = fn
        return fn
    return deco


@_handler("time")
def _time(_m: re.Match) -> str:
    now = datetime.now().astimezone()
    return now.strftime("It's %I:%M %p on %A, %B %d.").lstrip("0")


@_handler("sysinfo")
def _sysinfo(_m: re.Match) -> str:
    import platform
    import psutil

    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(str(_home()))
    return (
        f"{platform.system()} {platform.release()} · "
        f"CPU {psutil.cpu_percent(0.2):.0f}% · "
        f"RAM {vm.percent:.0f}% used · "
        f"Disk {disk.percent:.0f}% used ({disk.free/1024**3:.0f} GB free)"
    )


@_handler("screenshot")
def _screenshot(_m: re.Match) -> str:
    from pc_agent.tools.screen import take_screenshot

    r = take_screenshot({})
    if r.get("ok"):
        return f"Screenshot saved to {r.get('path')}"
    return f"Couldn't take a screenshot: {r.get('error')}"


@_handler("lock")
def _lock(_m: re.Match) -> str:
    if sys.platform == "win32":
        subprocess.Popen(["rundll32.exe", "user32.dll,LockWorkStation"])
        return "Locking the PC."
    return "Lock isn't wired on this OS yet."


@_handler("volume")
def _volume(m: re.Match) -> str:
    if sys.platform != "win32":
        return "Volume control is Windows-only for now."
    kind = (m.group("kind") or "").lower()
    # Use nircmd-free PowerShell / keybd via SendKeys isn't ideal; use Audio via PowerShell
    if "mute" in kind or "unmute" in kind:
        ps = (
            "(New-Object -ComObject WScript.Shell).SendKeys([char]173)"
        )
        subprocess.Popen(["powershell", "-NoProfile", "-Command", ps])
        return "Toggled mute."
    if "up" in kind or "louder" in kind:
        for _ in range(3):
            subprocess.Popen(
                ["powershell", "-NoProfile", "-Command",
                 "(New-Object -ComObject WScript.Shell).SendKeys([char]175)"]
            )
        return "Volume up."
    if "down" in kind or "quieter" in kind or "lower" in kind:
        for _ in range(3):
            subprocess.Popen(
                ["powershell", "-NoProfile", "-Command",
                 "(New-Object -ComObject WScript.Shell).SendKeys([char]174)"]
            )
        return "Volume down."
    return "Say volume up, volume down, or mute."


@_handler("open_url")
def _open_url(m: re.Match) -> str:
    url = m.group("url").strip()
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    webbrowser.open(url)
    return f"Opening {url}"


@_handler("open_folder")
def _open_folder(m: re.Match) -> str:
    name = (m.group("folder") or "").lower().strip()
    mapping = {
        "downloads": _home() / "Downloads",
        "download": _home() / "Downloads",
        "documents": _home() / "Documents",
        "docs": _home() / "Documents",
        "desktop": _home() / "Desktop",
        "pictures": _home() / "Pictures",
        "photos": _home() / "Pictures",
        "music": _home() / "Music",
        "videos": _home() / "Videos",
        "home": _home(),
    }
    path = mapping.get(name)
    if path is None:
        # treat as literal path
        path = Path(m.group("folder")).expanduser()
    if not path.exists():
        return f"I can't find {path}"
    return _open_path(path)


@_handler("open_app")
def _open_app(m: re.Match) -> str:
    app = (m.group("app") or "").lower().strip()
    win = {
        "explorer": ["explorer.exe"],
        "file explorer": ["explorer.exe"],
        "files": ["explorer.exe"],
        "chrome": ["cmd", "/c", "start", "", "chrome"],
        "google chrome": ["cmd", "/c", "start", "", "chrome"],
        "edge": ["cmd", "/c", "start", "", "msedge"],
        "firefox": ["cmd", "/c", "start", "", "firefox"],
        "notepad": ["notepad.exe"],
        "calculator": ["calc.exe"],
        "calc": ["calc.exe"],
        "settings": ["cmd", "/c", "start", "", "ms-settings:"],
        "cmd": ["cmd.exe"],
        "terminal": ["cmd", "/c", "start", "", "wt.exe"],
        "powershell": ["powershell.exe"],
        "paint": ["mspaint.exe"],
        "task manager": ["taskmgr.exe"],
        "taskmgr": ["taskmgr.exe"],
    }
    if sys.platform == "win32" and app in win:
        argv = win[app]
        if argv[0] == "cmd":
            subprocess.Popen(argv, shell=False)
        else:
            subprocess.Popen(argv)
        return f"Opening {app}."
    # generic
    try:
        if sys.platform == "win32":
            os.startfile(app)  # type: ignore[attr-defined]
        else:
            subprocess.Popen([app])
        return f"Opening {app}."
    except Exception as exc:
        return f"Couldn't open {app}: {exc}"


RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^(what('?s|\s+is)\s+the\s+)?time\b|\bwhat time is it\b", re.I), "time"),
    (re.compile(r"\b(system info|pc info|computer info|how.?s the (pc|system))\b", re.I), "sysinfo"),
    (re.compile(r"\b(take\s+a\s+)?screenshot\b|\bcapture (the )?screen\b", re.I), "screenshot"),
    (re.compile(r"\b(lock (the )?(pc|computer|screen)|lock workstation)\b", re.I), "lock"),
    (re.compile(r"\bvolume\s+(?P<kind>up|down|mute|unmute|louder|quieter|lower)\b", re.I), "volume"),
    (re.compile(r"\b(mute|unmute)(\s+volume)?\b", re.I), "volume"),
    (re.compile(r"\bopen\s+(?P<url>https?://\S+|\S+\.(com|org|net|io|dev|ai)\S*)\b", re.I), "open_url"),
    (re.compile(
        r"\b(open|show)\s+(?P<folder>downloads?|documents|docs|desktop|pictures|photos|music|videos|home)\b",
        re.I,
    ), "open_folder"),
    (re.compile(
        r"\b(open|launch|start)\s+(?P<app>file explorer|explorer|files|google chrome|chrome|edge|"
        r"firefox|notepad|calculator|calc|settings|cmd|terminal|powershell|paint|"
        r"task manager|taskmgr)\b",
        re.I,
    ), "open_app"),
    (re.compile(r"\b(open|show)\s+(?P<folder>[A-Za-z]:\\[^\n]+|~\/[^\n]+)\b", re.I), "open_folder"),
]


def try_handle(text: str) -> str | None:
    """Return a reply string if handled offline, else None."""
    line = (text or "").strip()
    if not line or line.startswith("/"):
        return None
    for pat, name in RULES:
        m = pat.search(line)
        if not m:
            continue
        fn = HANDLERS.get(name)
        if not fn:
            continue
        # Special-case mute patterns into volume with kind=
        if name == "volume" and "kind" not in m.re.groupindex:
            class _M:
                def group(self, k, default=None):
                    if k == "kind":
                        return "mute" if "mute" in line.lower() else "up"
                    return m.group(k) if k in (m.re.groupindex or {}) else default
            return fn(_M())  # type: ignore[arg-type]
        try:
            return fn(m)
        except Exception as exc:
            return f"Tried offline, but hit: {exc}"
    return None
