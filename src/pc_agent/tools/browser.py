"""Browser automation via Playwright. Optional Chrome profile with consent."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from pc_agent.config import load_config
from pc_agent.core import safety

logger = logging.getLogger("pc_agent.browser")

DECLARATIONS = [
    {
        "name": "browser_navigate",
        "description": (
            "Open a URL in an automated browser (Playwright). "
            "Set use_profile=true only with user consent to reuse their Chrome profile."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "use_profile": {
                    "type": "boolean",
                    "description": "Use the user's Chrome profile (requires confirmation).",
                },
                "headless": {"type": "boolean"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "browser_click",
        "description": "Click a CSS selector in the current browser page.",
        "parameters": {
            "type": "object",
            "properties": {"selector": {"type": "string"}},
            "required": ["selector"],
        },
    },
    {
        "name": "browser_type",
        "description": "Type into a CSS selector in the current browser page.",
        "parameters": {
            "type": "object",
            "properties": {
                "selector": {"type": "string"},
                "text": {"type": "string"},
            },
            "required": ["selector", "text"],
        },
    },
    {
        "name": "browser_snapshot",
        "description": "Return the current page title, URL, and a text snapshot (accessibility tree truncated).",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "browser_close",
        "description": "Close the automated browser session.",
        "parameters": {"type": "object", "properties": {}},
    },
]

_state: dict = {"pw": None, "browser": None, "context": None, "page": None}


def _chrome_user_data() -> Path | None:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "User Data"
        return base if base.exists() else None
    # Linux
    p = Path.home() / ".config" / "google-chrome"
    return p if p.exists() else None


def _ensure_page(use_profile: bool = False, headless: bool = False):
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise RuntimeError(
            f"Playwright not installed ({exc}). Run: pip install playwright && playwright install chromium"
        ) from exc

    if _state["page"] is not None:
        return _state["page"]

    pw = sync_playwright().start()
    _state["pw"] = pw
    if use_profile:
        if not safety.confirm(
            "browser_profile",
            "Use your Chrome profile (cookies/logins) for browser automation?",
        ):
            pw.stop()
            _state["pw"] = None
            raise RuntimeError("User denied Chrome profile use")
        ud = _chrome_user_data()
        if ud is None:
            pw.stop()
            _state["pw"] = None
            raise RuntimeError("Chrome user data folder not found")
        context = pw.chromium.launch_persistent_context(
            user_data_dir=str(ud),
            headless=headless,
            channel="chrome",
            args=["--disable-blink-features=AutomationControlled"],
        )
        _state["context"] = context
        page = context.pages[0] if context.pages else context.new_page()
        _state["page"] = page
        return page

    browser = pw.chromium.launch(headless=headless)
    _state["browser"] = browser
    context = browser.new_context()
    _state["context"] = context
    page = context.new_page()
    _state["page"] = page
    return page


def browser_navigate(args: dict) -> dict:
    if safety.is_halted():
        return {"ok": False, "error": "Agent is halted (kill-switch)."}
    url = args.get("url") or ""
    use_profile = bool(args.get("use_profile") or load_config().get("browser_use_chrome_profile"))
    headless = bool(args.get("headless", False))
    try:
        page = _ensure_page(use_profile=use_profile, headless=headless)
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        safety.log_action("browser", "navigate", {"url": url, "profile": use_profile})
        return {"ok": True, "url": page.url, "title": page.title()}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def browser_click(args: dict) -> dict:
    if safety.is_halted():
        return {"ok": False, "error": "Agent is halted (kill-switch)."}
    if _state["page"] is None:
        return {"ok": False, "error": "no browser session; call browser_navigate first"}
    try:
        _state["page"].click(args["selector"], timeout=15000)
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def browser_type(args: dict) -> dict:
    if safety.is_halted():
        return {"ok": False, "error": "Agent is halted (kill-switch)."}
    if _state["page"] is None:
        return {"ok": False, "error": "no browser session"}
    try:
        _state["page"].fill(args["selector"], args.get("text") or "")
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def browser_snapshot(_args: dict) -> dict:
    if _state["page"] is None:
        return {"ok": False, "error": "no browser session"}
    try:
        page = _state["page"]
        # accessibility snapshot if available
        snap = ""
        try:
            snap = str(page.locator("body").inner_text(timeout=5000))[:4000]
        except Exception:
            snap = ""
        return {"ok": True, "url": page.url, "title": page.title(), "text": snap}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def browser_close(_args: dict) -> dict:
    try:
        if _state["context"] is not None:
            _state["context"].close()
        if _state["browser"] is not None:
            _state["browser"].close()
        if _state["pw"] is not None:
            _state["pw"].stop()
    except Exception as exc:
        logger.debug("browser close: %s", exc)
    _state.update({"pw": None, "browser": None, "context": None, "page": None})
    return {"ok": True}


_HANDLERS = {
    "browser_navigate": browser_navigate,
    "browser_click": browser_click,
    "browser_type": browser_type,
    "browser_snapshot": browser_snapshot,
    "browser_close": browser_close,
}


def make_handler(name: str, **_kwargs):
    return _HANDLERS[name]
