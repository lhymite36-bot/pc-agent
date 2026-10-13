"""Paths, defaults, and runtime settings for PC Agent."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

APP_NAME = "PC Agent"
APP_ID = "pc-agent"
VERSION = "1.0.6"

# --- Brain ---
DEFAULT_MODEL = "gemini-3.6-flash"
FALLBACK_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_FALLBACKS = ["gemini-embedding-001", "text-embedding-004"]

# --- Voice (realistic / human by default) ---
# Live native-audio is the default path (natural prosody, emotion, breaths, barge-in).
LIVE_MODEL = "gemini-3.8-live"
LIVE_MODEL_FALLBACKS = [
    "gemini-3.8-live",
    "gemini-2.5-flash-native-audio-preview-12-2025",
    "gemini-3.1-flash-live-preview",
]
# Best expressive TTS for non-live replies (never the lite/flat models).
TTS_MODEL = "gemini-3.8-flash-tts"
TTS_MODEL_FALLBACKS = [
    "gemini-3.8-flash-tts",
    "gemini-2.5-flash-preview-tts",
]
# Warm natural default; picker exposes the rest with previews.
DEFAULT_VOICE = "Sulafat"  # Warm
VOICE_CATALOG = [
    {"id": "Sulafat", "label": "Sulafat — Warm", "style": "warm"},
    {"id": "Aoede", "label": "Aoede — Breezy", "style": "breezy"},
    {"id": "Achird", "label": "Achird — Friendly", "style": "friendly"},
    {"id": "Callirrhoe", "label": "Callirrhoe — Easy-going", "style": "easy-going"},
    {"id": "Puck", "label": "Puck — Upbeat", "style": "upbeat"},
    {"id": "Kore", "label": "Kore — Firm", "style": "firm"},
    {"id": "Charon", "label": "Charon — Informative", "style": "informative"},
    {"id": "Fenrir", "label": "Fenrir — Excitable", "style": "excitable"},
    {"id": "Enceladus", "label": "Enceladus — Breathy", "style": "breathy"},
    {"id": "Zephyr", "label": "Zephyr — Bright", "style": "bright"},
    {"id": "Vindemiatrix", "label": "Vindemiatrix — Gentle", "style": "gentle"},
    {"id": "Algieba", "label": "Algieba — Smooth", "style": "smooth"},
]
# Short style so replies don't sound like reading a script.
VOICE_STYLE_PROMPT = (
    "Speak conversationally and casually. Use contractions (I'm, you're, that's, won't). "
    "Keep sentences short. Sound like a sharp friend helping at a desk — warm, clear, "
    "natural pacing with light emotion. Never sound like you're reading aloud."
)

CHAT_HOST = "127.0.0.1"
CHAT_PORT = 8765

KILL_SWITCH_HOTKEY = "<ctrl>+<alt>+<esc>"

CONFIRM_CATEGORIES = frozenset(
    {
        "delete_file",
        "install_software",
        "send_message",
        "send_email",
        "payment",
        "registry_edit",
        "shutdown",
        "browser_profile",  # using the user's Chrome profile
    }
)

DEFAULT_CONFIG = {
    "model": DEFAULT_MODEL,
    "agent_name": "Atlas",
    "personality": (
        "Witty, warm, and sharp. Light sarcasm OK, never cruel. "
        "Keep replies short unless the user wants detail. You're better at "
        "getting stuff done on this PC than a chatbox — prove it."
    ),
    "voice_mode": "live",  # live | tts | offline
    "voice_id": DEFAULT_VOICE,
    "voice_enabled": True,
    "telegram_token": "",
    "telegram_allowlist": [],
    "autostart": True,
    "confirm_dangerous": True,
    "proactive": True,
    "screen_awareness": False,  # privacy: off by default
    "screen_awareness_interval_sec": 45,
    "browser_use_chrome_profile": False,
}


def data_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    path = base / APP_ID
    path.mkdir(parents=True, exist_ok=True)
    return path


def skills_dir() -> Path:
    path = data_dir() / "skills"
    path.mkdir(parents=True, exist_ok=True)
    return path


def plugins_dir() -> Path:
    path = data_dir() / "plugins"
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_path() -> Path:
    return data_dir() / "config.json"


def db_path() -> Path:
    return data_dir() / "memory.db"


def log_path() -> Path:
    return data_dir() / "actions.log"


def load_config() -> dict:
    path = config_path()
    cfg = dict(DEFAULT_CONFIG)
    if path.exists():
        with path.open("r", encoding="utf-8-sig") as f:
            cfg.update(json.load(f))
    return cfg


def save_config(cfg: dict) -> None:
    merged = dict(DEFAULT_CONFIG)
    merged.update(cfg)
    with config_path().open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)


def get_api_key() -> str | None:
    try:
        import keyring

        return keyring.get_password(APP_ID, "gemini_api_key")
    except Exception:
        fallback = data_dir() / ".api_key"
        if fallback.exists():
            return fallback.read_text(encoding="utf-8").strip() or None
        return None


def set_api_key(key: str) -> None:
    try:
        import keyring

        keyring.set_password(APP_ID, "gemini_api_key", key)
        return
    except Exception:
        pass
    fallback = data_dir() / ".api_key"
    fallback.write_text(key.strip(), encoding="utf-8")
    try:
        os.chmod(fallback, 0o600)
    except OSError:
        pass


def clear_api_key() -> None:
    try:
        import keyring

        keyring.delete_password(APP_ID, "gemini_api_key")
    except Exception:
        pass
    fallback = data_dir() / ".api_key"
    if fallback.exists():
        fallback.unlink()
