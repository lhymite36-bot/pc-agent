"""Open apps and URLs."""

from __future__ import annotations

import os
import subprocess
import sys
import webbrowser

from pc_agent.core import safety

DECLARATIONS = [
    {
        "name": "open_url",
        "description": "Open a URL in the default browser.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "open_app",
        "description": (
            "Launch an application. On Windows, pass an executable name/path "
            "(e.g. notepad.exe) or a Start Menu app name."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional arguments.",
                },
            },
            "required": ["target"],
        },
    },
]


def open_url(args: dict) -> dict:
    if safety.is_halted():
        return {"ok": False, "error": "Agent is halted (kill-switch)."}
    url = args.get("url") or ""
    if not url.startswith(("http://", "https://", "file://")):
        url = "https://" + url
    try:
        webbrowser.open(url)
        safety.log_action("apps", "open_url", {"url": url})
        return {"ok": True, "url": url}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def open_app(args: dict) -> dict:
    if safety.is_halted():
        return {"ok": False, "error": "Agent is halted (kill-switch)."}
    target = args.get("target") or ""
    extra = list(args.get("args") or [])
    try:
        if sys.platform == "win32":
            # os.startfile for files; subprocess for exes with args
            if not extra and (os.path.isfile(target) or target.lower().endswith((".exe", ".lnk", ".bat"))):
                os.startfile(target)  # type: ignore[attr-defined]
            else:
                subprocess.Popen([target, *extra], shell=False)
        else:
            subprocess.Popen([target, *extra])
        safety.log_action("apps", "open_app", {"target": target})
        return {"ok": True, "target": target}
    except Exception as exc:
        # Fallback: shell out
        try:
            if sys.platform == "win32":
                subprocess.Popen(["cmd", "/c", "start", "", target, *extra])
            else:
                subprocess.Popen(["xdg-open", target])
            return {"ok": True, "target": target, "via": "fallback"}
        except Exception as exc2:
            return {"ok": False, "error": f"{exc}; fallback: {exc2}"}


_HANDLERS = {"open_url": open_url, "open_app": open_app}


def make_handler(name: str, **_kwargs):
    return _HANDLERS[name]
