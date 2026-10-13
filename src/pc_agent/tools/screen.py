"""Screenshots + vision helper (bytes handed to the brain)."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from pc_agent.config import data_dir
from pc_agent.core import safety

DECLARATIONS = [
    {
        "name": "take_screenshot",
        "description": (
            "Capture the screen (or a region) as a PNG. Returns a path and a short "
            "base64 preview. Use this before describing what is on screen."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "monitor": {
                    "type": "integer",
                    "description": "Monitor index (1-based). Default 1.",
                },
                "region": {
                    "type": "object",
                    "description": "Optional {left, top, width, height}.",
                    "properties": {
                        "left": {"type": "integer"},
                        "top": {"type": "integer"},
                        "width": {"type": "integer"},
                        "height": {"type": "integer"},
                    },
                },
            },
        },
    }
]


def take_screenshot(args: dict) -> dict:
    if safety.is_halted():
        return {"ok": False, "error": "Agent is halted (kill-switch)."}
    try:
        import mss
        from PIL import Image
    except Exception as exc:
        return {"ok": False, "error": f"screenshot deps missing: {exc}"}

    monitor_i = int(args.get("monitor") or 1)
    region = args.get("region") or None
    try:
        with mss.mss() as sct:
            if region:
                mon = {
                    "left": int(region["left"]),
                    "top": int(region["top"]),
                    "width": int(region["width"]),
                    "height": int(region["height"]),
                }
            else:
                monitors = sct.monitors
                if monitor_i < 1 or monitor_i >= len(monitors):
                    monitor_i = 1 if len(monitors) > 1 else 0
                mon = monitors[monitor_i]
            shot = sct.grab(mon)
            img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        out = data_dir() / "last_screenshot.png"
        img.save(out, format="PNG")
        # Small preview for tool result (full image is on disk for vision)
        thumb = img.copy()
        thumb.thumbnail((640, 360))
        buf = io.BytesIO()
        thumb.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        safety.log_action("screen", "screenshot", {"path": str(out)})
        return {
            "ok": True,
            "path": str(out),
            "width": img.width,
            "height": img.height,
            "preview_png_base64": b64[:20000],
            "note": "Full PNG saved at path. Attach it for vision in the next turn if needed.",
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def load_last_screenshot_bytes() -> bytes | None:
    path = data_dir() / "last_screenshot.png"
    if path.exists():
        return path.read_bytes()
    return None


def make_handler(name: str, **_kwargs):
    if name == "take_screenshot":
        return take_screenshot
    raise KeyError(name)
