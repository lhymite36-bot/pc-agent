"""Mouse and keyboard control via pyautogui."""

from __future__ import annotations

from pc_agent.core import safety

DECLARATIONS = [
    {
        "name": "mouse_move",
        "description": "Move the mouse cursor to screen coordinates.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "duration": {"type": "number", "description": "Seconds to move (default 0.2)."},
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "mouse_click",
        "description": "Click the mouse. Optionally move first.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "button": {"type": "string", "enum": ["left", "right", "middle"]},
                "clicks": {"type": "integer"},
            },
        },
    },
    {
        "name": "type_text",
        "description": "Type text with the keyboard (into the focused window).",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "interval": {"type": "number", "description": "Delay between keys."},
            },
            "required": ["text"],
        },
    },
    {
        "name": "hotkey",
        "description": "Press a key combination, e.g. keys=['ctrl','c'].",
        "parameters": {
            "type": "object",
            "properties": {
                "keys": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["keys"],
        },
    },
]


def _pg():
    import pyautogui

    pyautogui.FAILSAFE = True
    return pyautogui


def mouse_move(args: dict) -> dict:
    if safety.is_halted():
        return {"ok": False, "error": "Agent is halted (kill-switch)."}
    pg = _pg()
    x, y = int(args["x"]), int(args["y"])
    dur = float(args.get("duration") or 0.2)
    pg.moveTo(x, y, duration=dur)
    safety.log_action("input", "mouse_move", {"x": x, "y": y})
    return {"ok": True, "x": x, "y": y}


def mouse_click(args: dict) -> dict:
    if safety.is_halted():
        return {"ok": False, "error": "Agent is halted (kill-switch)."}
    pg = _pg()
    button = args.get("button") or "left"
    clicks = int(args.get("clicks") or 1)
    if "x" in args and "y" in args:
        pg.click(int(args["x"]), int(args["y"]), clicks=clicks, button=button)
    else:
        pg.click(clicks=clicks, button=button)
    safety.log_action("input", "mouse_click", {"button": button, "clicks": clicks})
    return {"ok": True}


def type_text(args: dict) -> dict:
    if safety.is_halted():
        return {"ok": False, "error": "Agent is halted (kill-switch)."}
    text = args.get("text") or ""
    # Guard: don't type secrets into unknown fields without confirmation for send-ish text
    if any(w in text.lower() for w in ("password", "ssn", "credit card")):
        if not safety.confirm("send_message", f"Type sensitive-looking text ({len(text)} chars)?"):
            return {"ok": False, "error": "User denied confirmation"}
    pg = _pg()
    interval = float(args.get("interval") or 0.02)
    pg.typewrite(text, interval=interval) if text.isascii() else pg.write(text, interval=interval)
    safety.log_action("input", "type_text", {"chars": len(text)})
    return {"ok": True, "chars": len(text)}


def hotkey(args: dict) -> dict:
    if safety.is_halted():
        return {"ok": False, "error": "Agent is halted (kill-switch)."}
    keys = list(args.get("keys") or [])
    if not keys:
        return {"ok": False, "error": "no keys"}
    pg = _pg()
    pg.hotkey(*keys)
    safety.log_action("input", "hotkey", {"keys": keys})
    return {"ok": True, "keys": keys}


_HANDLERS = {
    "mouse_move": mouse_move,
    "mouse_click": mouse_click,
    "type_text": type_text,
    "hotkey": hotkey,
}


def make_handler(name: str, **_kwargs):
    return _HANDLERS[name]
