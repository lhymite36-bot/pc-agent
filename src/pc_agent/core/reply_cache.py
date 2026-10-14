"""Tiny reply cache: answers to general questions keep working at zero quota."""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

from pc_agent.config import data_dir

MAX_ENTRIES = 400
TTL_SEC = 30 * 24 * 3600
_QUESTION = re.compile(r"^(what|who|why|how|when|where|which|explain|define|tell me|is |are |can |does |do )", re.I)
_VOLATILE = re.compile(r"\b(time|today|now|weather|news|price|latest|screen|this|my |open|file|folder)\b", re.I)
_lock = threading.Lock()


def _path() -> Path:
    return data_dir() / "reply_cache.json"


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", (text or "").lower()).strip()


def cacheable(text: str, reply: str) -> bool:
    t = (text or "").strip()
    if not reply or len(reply) < 2 or not (_QUESTION.match(t) or t.endswith("?")):
        return False
    if _VOLATILE.search(t):
        return False
    low = reply.lower()
    return not any(s in low for s in ("quota", "busy", "something went wrong", "(no text)"))


def _load() -> dict:
    try:
        return json.loads(_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def put(text: str, reply: str) -> bool:
    if not cacheable(text, reply):
        return False
    with _lock:
        data = _load()
        data[norm(text)] = {"r": reply[:4000], "t": time.time()}
        if len(data) > MAX_ENTRIES:
            for k, _ in sorted(data.items(), key=lambda kv: kv[1]["t"])[: len(data) - MAX_ENTRIES]:
                data.pop(k, None)
        _path().parent.mkdir(parents=True, exist_ok=True)
        _path().write_text(json.dumps(data), encoding="utf-8")
    return True


def get(text: str) -> str | None:
    e = _load().get(norm(text))
    if e and time.time() - e.get("t", 0) < TTL_SEC:
        return e["r"]
    return None
