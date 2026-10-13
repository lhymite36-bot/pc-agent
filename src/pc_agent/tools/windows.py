"""Window list / focus via pygetwindow (best effort on non-Windows)."""

from __future__ import annotations

from pc_agent.core import safety

DECLARATIONS = [
    {
        "name": "list_windows",
        "description": "List open window titles.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "focus_window",
        "description": "Bring a window to the foreground by title substring.",
        "parameters": {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        },
    },
]


def list_windows(_args: dict) -> dict:
    try:
        import pygetwindow as gw
    except Exception as exc:
        return {"ok": False, "error": f"pygetwindow unavailable: {exc}"}
    titles = []
    try:
        for w in gw.getAllWindows():
            t = (w.title or "").strip()
            if t:
                titles.append(
                    {
                        "title": t,
                        "left": getattr(w, "left", None),
                        "top": getattr(w, "top", None),
                        "width": getattr(w, "width", None),
                        "height": getattr(w, "height", None),
                    }
                )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "windows": titles[:100]}


def focus_window(args: dict) -> dict:
    if safety.is_halted():
        return {"ok": False, "error": "Agent is halted (kill-switch)."}
    try:
        import pygetwindow as gw
    except Exception as exc:
        return {"ok": False, "error": f"pygetwindow unavailable: {exc}"}
    needle = (args.get("title") or "").lower()
    try:
        matches = [w for w in gw.getAllWindows() if needle in (w.title or "").lower()]
        if not matches:
            return {"ok": False, "error": "no matching window"}
        w = matches[0]
        try:
            w.activate()
        except Exception:
            # Some platforms need restore first
            try:
                w.restore()
                w.activate()
            except Exception as exc:
                return {"ok": False, "error": str(exc)}
        safety.log_action("windows", "focus", {"title": w.title})
        return {"ok": True, "title": w.title}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


_HANDLERS = {"list_windows": list_windows, "focus_window": focus_window}


def make_handler(name: str, **_kwargs):
    return _HANDLERS[name]
