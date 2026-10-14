"""Optional free local LLM via Ollama (localhost:11434). Not bundled."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger("pc_agent.ollama")

DEFAULT_HOST = "http://127.0.0.1:11434"
PREFERRED_MODELS = ["llama3.2:3b", "qwen2.5:3b", "llama3.2", "phi3:mini", "gemma2:2b"]


_probe_cache: dict[str, tuple[float, bool, list[str]]] = {}
PROBE_TTL_SEC = 60.0


def available(host: str = DEFAULT_HOST, timeout: float = 0.8) -> tuple[bool, list[str]]:
    """Fast probe (<=1 s), cached 60 s, never raises. Not installed => (False, [])."""
    import time as _t
    hit = _probe_cache.get(host)
    if hit and _t.time() - hit[0] < PROBE_TTL_SEC:
        return hit[1], hit[2]
    ok, models = _probe(host, min(timeout, 1.0))
    _probe_cache[host] = (_t.time(), ok, models)
    return ok, models


def _probe(host: str, timeout: float) -> tuple[bool, list[str]]:
    try:
        r = httpx.get(f"{host}/api/tags", timeout=timeout)
        r.raise_for_status()
        models = [m.get("name", "") for m in (r.json().get("models") or [])]
        return True, models
    except Exception as exc:
        logger.debug("ollama not available: %s", exc)
        return False, []


def pick_model(installed: list[str]) -> str | None:
    names = installed or []
    for pref in PREFERRED_MODELS:
        for n in names:
            if n == pref or n.startswith(pref + ":") or pref in n:
                return n
    return names[0] if names else None


def chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    host: str = DEFAULT_HOST,
    tools: list[dict] | None = None,
) -> str:
    """Simple chat. tools are described in the system prompt (Ollama tool calling varies)."""
    ok, installed = available(host)
    if not ok:
        raise RuntimeError("Ollama is not running at localhost:11434")
    model = model or pick_model(installed)
    if not model:
        raise RuntimeError(
            "Ollama is running but has no models. Try: ollama pull llama3.2:3b"
        )
    system_extra = ""
    if tools:
        names = ", ".join(t.get("name", "?") for t in tools[:40])
        system_extra = (
            f"\nYou can suggest tools by name from: {names}. "
            "For offline-only sessions, prefer answering directly."
        )
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.4, "num_predict": 512},
    }
    # prepend system nudge into first system message if present
    if system_extra and payload["messages"]:
        if payload["messages"][0].get("role") == "system":
            payload["messages"][0]["content"] += system_extra
        else:
            payload["messages"].insert(0, {"role": "system", "content": system_extra.strip()})
    r = httpx.post(f"{host}/api/chat", json=payload, timeout=httpx.Timeout(45.0, connect=1.0))
    r.raise_for_status()
    data = r.json()
    return ((data.get("message") or {}).get("content") or "").strip()
