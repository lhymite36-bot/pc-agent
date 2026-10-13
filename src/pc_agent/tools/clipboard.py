"""Clipboard get/set."""

from __future__ import annotations

from pc_agent.core import safety

DECLARATIONS = [
    {
        "name": "clipboard_get",
        "description": "Read the current clipboard text.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "clipboard_set",
        "description": "Set the clipboard text.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
]


def clipboard_get(_args: dict) -> dict:
    try:
        import pyperclip

        text = pyperclip.paste()
        return {"ok": True, "text": text}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def clipboard_set(args: dict) -> dict:
    if safety.is_halted():
        return {"ok": False, "error": "Agent is halted (kill-switch)."}
    try:
        import pyperclip

        text = args.get("text") or ""
        pyperclip.copy(text)
        safety.log_action("clipboard", "set", {"chars": len(text)})
        return {"ok": True, "chars": len(text)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


_HANDLERS = {"clipboard_get": clipboard_get, "clipboard_set": clipboard_set}


def make_handler(name: str, **_kwargs):
    return _HANDLERS[name]
