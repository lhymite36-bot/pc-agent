"""Free, no-card LLM providers (OpenAI-compatible) with per-key cooldowns.

Used when Gemini is out of quota / unavailable. Never suggests paying.
Verified Oct 2026:
  - Groq: free plan, no card. openai/gpt-oss-120b, qwen/qwen3.8-27b, gpt-oss-20b;
    whisper-large-v3-turbo for speech-to-text. (llama-3.3-70b shut down Aug 2026.)
  - OpenRouter: ':free' models; 'openrouter/free' auto-routes to a free
    tool-capable model. No card.
  - Mistral: Free mode (default), no card. mistral-small-latest.
  - Cloudflare Workers AI: free daily allowance, no card. Key = "ACCOUNT_ID:TOKEN".
  - Cerebras: works if you already have a key, but new accounts now need a
    payment method, so the setup wizard does NOT recommend it.
  - GitHub Models: retired 30 Jul 2026 — not supported.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

from pc_agent.config import data_dir

logger = logging.getLogger("pc_agent.providers")

ATTEMPT_TIMEOUT_SEC = 8.0     # each provider gets at most this; failover well under 10 s
CONNECT_TIMEOUT_SEC = 3.0
MAX_TOOL_ROUNDS = 4


@dataclass
class Provider:
    name: str
    label: str
    base_url: str
    models: list[str]
    signup: str
    recommended: bool = True      # free & no card
    note: str = ""
    stt_model: str | None = None
    headers: dict[str, str] = field(default_factory=dict)

    def url(self, key: str, path: str) -> str:
        base = self.base_url
        if "{account}" in base:
            acct = key.split(":", 1)[0]
            base = base.replace("{account}", acct)
        return base.rstrip("/") + path

    @staticmethod
    def token(key: str, name: str) -> str:
        return key.split(":", 1)[1] if name == "cloudflare" and ":" in key else key


# Ranked by speed (fastest first)
PROVIDERS: list[Provider] = [
    Provider("groq", "Groq", "https://api.groq.com/openai/v1",
             ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"],
             "https://console.groq.com/keys", stt_model="whisper-large-v3-turbo",
             note="fastest; free plan ~1000 requests/day, also free speech-to-text"),
    Provider("cerebras", "Cerebras", "https://api.cerebras.ai/v1",
             ["gpt-oss-120b", "qwen-3.8-27b"],
             "https://cloud.cerebras.ai", recommended=False,
             note="very fast, but new accounts now need a card — skip unless you already have a key"),
    Provider("openrouter", "OpenRouter (free models)", "https://openrouter.ai/api/v1",
             ["openrouter/free", "nvidia/nemotron-3-ultra-550b-a55b:free", "google/gemma-4-26b-a4b-it:free"],
             "https://openrouter.ai/keys",
             headers={"HTTP-Referer": "https://github.com/lhymite36-bot/pc-agent", "X-Title": "Atlas"},
             note="only ':free' models are used"),
    Provider("mistral", "Mistral (Free mode)", "https://api.mistral.ai/v1",
             ["mistral-small-latest"], "https://console.mistral.ai/api-keys",
             note="free mode, no card"),
    Provider("cloudflare", "Cloudflare Workers AI", "https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1",
             ["@cf/openai/gpt-oss-120b", "@cf/meta/llama-3.3-70b-instruct-fp8-fast"],
             "https://dash.cloudflare.com/profile/api-tokens",
             note='free daily allowance; add as "ACCOUNT_ID:API_TOKEN"'),
]
BY_NAME = {p.name: p for p in PROVIDERS}
ALIASES = {"or": "openrouter", "cf": "cloudflare", "workers": "cloudflare", "gemini": "gemini"}


def resolve(name: str) -> str | None:
    n = (name or "").strip().lower()
    n = ALIASES.get(n, n)
    return n if n in BY_NAME or n == "gemini" else None


# --- key store + cooldowns (persisted) ---

_lock = threading.Lock()


def _keys_path() -> Path:
    return data_dir() / "free_keys.json"


def _health_path() -> Path:
    return data_dir() / "provider_health.json"


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def list_keys(provider: str) -> list[str]:
    return list(_read(_keys_path()).get(provider, []))


def add_key(provider: str, key: str) -> int:
    key = key.strip()
    with _lock:
        data = _read(_keys_path())
        keys = data.setdefault(provider, [])
        if key not in keys:
            keys.append(key)
        _write(_keys_path(), data)
        return len(keys)


def remove_key(provider: str, index: int) -> bool:
    with _lock:
        data = _read(_keys_path())
        keys = data.get(provider, [])
        if 0 <= index < len(keys):
            keys.pop(index)
            _write(_keys_path(), data)
            return True
        return False


def configured() -> list[str]:
    data = _read(_keys_path())
    return [p.name for p in PROVIDERS if data.get(p.name)]


def _slot(provider: str, idx: int) -> str:
    return f"{provider}#{idx}"


def cooling_until(provider: str, idx: int, now: float | None = None) -> float:
    until = float(_read(_health_path()).get(_slot(provider, idx), 0))
    return until if until > (now or time.time()) else 0.0


def mark_cooling(provider: str, idx: int, seconds: float) -> None:
    with _lock:
        data = _read(_health_path())
        data[_slot(provider, idx)] = time.time() + max(5.0, seconds)
        _write(_health_path(), data)


def next_reset(provider: str) -> float:
    times = [cooling_until(provider, i) for i in range(len(list_keys(provider)))]
    times = [t for t in times if t]
    return min(times) if times else 0.0


def retry_after(resp: httpx.Response | None, body: str = "") -> float:
    if resp is not None:
        for h in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
            v = resp.headers.get(h)
            if v:
                secs = _parse_dur(v)
                if secs:
                    return secs
    m = re.search(r"(?:try again|retry) in ([0-9hms.]+)", body or "", re.I)
    if m:
        return _parse_dur(m.group(1)) or 60.0
    return 60.0


def _parse_dur(v: str) -> float:
    v = v.strip()
    try:
        return float(v)
    except ValueError:
        pass
    total, found = 0.0, False
    for num, unit in re.findall(r"([0-9.]+)\s*(h|ms|m|s)", v):
        found = True
        total += float(num) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
    return total if found else 0.0


# --- tool schema conversion ---

def to_openai_tools(declarations: list[dict], allow: set[str] | None = None) -> list[dict]:
    out = []
    for d in declarations or []:
        name = d.get("name")
        if not name or name == "speak" or (allow is not None and name not in allow):
            continue
        params = _lower_types(d.get("parameters") or {"type": "object", "properties": {}})
        out.append({"type": "function", "function": {
            "name": name, "description": (d.get("description") or "")[:300], "parameters": params}})
    return out


def _lower_types(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {k: (v.lower() if k == "type" and isinstance(v, str) else _lower_types(v)) for k, v in schema.items()}
    if isinstance(schema, list):
        return [_lower_types(x) for x in schema]
    return schema


# --- chat with failover ---

class ProvidersExhausted(Exception):
    pass


Post = Callable[..., httpx.Response]


def _default_post(url: str, **kw) -> httpx.Response:
    return httpx.post(url, **kw)


def candidates(now: float | None = None) -> list[tuple[Provider, int, str]]:
    out = []
    for p in PROVIDERS:
        for i, k in enumerate(list_keys(p.name)):
            if not cooling_until(p.name, i, now):
                out.append((p, i, k))
    return out


def chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    handlers: dict[str, Callable[[dict], Any]] | None = None,
    post: Post | None = None,
    on_provider: Callable[[str, str], None] | None = None,
) -> str:
    """Run a tool-calling chat across free providers. Returns final text."""
    post = post or _default_post
    cands = candidates()
    if not cands:
        raise ProvidersExhausted("no free provider keys available")
    last = ""
    for prov, idx, key in cands:
        models = list(prov.models)
        for model in models:
            try:
                text = _run(prov, key, model, list(messages), tools, handlers or {}, post)
                if on_provider:
                    on_provider(prov.name, model)
                return text
            except _ModelGone:
                continue  # try next model on same provider
            except _Cool as c:
                mark_cooling(prov.name, idx, c.seconds)
                last = f"{prov.name}: {c.reason}"
                logger.info("provider %s#%d cooling %.0fs (%s)", prov.name, idx, c.seconds, c.reason)
                break
    raise ProvidersExhausted(last or "all free providers failed")


class _Cool(Exception):
    def __init__(self, seconds: float, reason: str) -> None:
        super().__init__(reason)
        self.seconds, self.reason = seconds, reason


class _ModelGone(Exception):
    pass


def _run(prov, key, model, messages, tools, handlers, post) -> str:
    headers = {"Authorization": f"Bearer {Provider.token(key, prov.name)}", **prov.headers}
    timeout = httpx.Timeout(ATTEMPT_TIMEOUT_SEC, connect=CONNECT_TIMEOUT_SEC)
    for _ in range(MAX_TOOL_ROUNDS + 1):
        body: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0.4, "max_tokens": 800}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        try:
            r = post(prov.url(key, "/chat/completions"), json=body, headers=headers, timeout=timeout)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise _Cool(60, f"network/timeout: {type(exc).__name__}") from exc
        if r.status_code == 429:
            raise _Cool(retry_after(r, r.text), "quota/rate limit")
        if r.status_code in (401, 403):
            raise _Cool(6 * 3600, f"key rejected ({r.status_code})")
        if r.status_code in (404,) or (r.status_code == 400 and "model" in r.text.lower() and "tool" not in r.text.lower()):
            raise _ModelGone(model)
        if r.status_code >= 500:
            raise _Cool(30, f"server {r.status_code}")
        if r.status_code >= 400:
            raise _Cool(120, f"http {r.status_code}")
        data = r.json()
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        calls = msg.get("tool_calls") or []
        if not calls:
            return (msg.get("content") or "").strip() or "(no text)"
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for c in calls:
            fn = c.get("function") or {}
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            h = handlers.get(name)
            try:
                result = h(args) if h else {"ok": False, "error": f"unknown tool {name}"}
            except Exception as exc:
                result = {"ok": False, "error": str(exc)}
            s = json.dumps(result, default=str)
            messages.append({"role": "tool", "tool_call_id": c.get("id", name), "content": s[:3000]})
    return "Stopped after too many tool steps."


# --- speech-to-text (Groq Whisper, free) ---

def transcribe(wav_bytes: bytes, post: Post | None = None) -> str:
    post = post or _default_post
    prov = BY_NAME["groq"]
    for i, key in enumerate(list_keys("groq")):
        if cooling_until("groq", i):
            continue
        try:
            r = post(prov.url(key, "/audio/transcriptions"),
                     headers={"Authorization": f"Bearer {key}"},
                     files={"file": ("speech.wav", wav_bytes, "audio/wav")},
                     data={"model": prov.stt_model, "response_format": "json"},
                     timeout=httpx.Timeout(ATTEMPT_TIMEOUT_SEC, connect=CONNECT_TIMEOUT_SEC))
        except Exception:
            mark_cooling("groq", i, 60)
            continue
        if r.status_code == 429:
            mark_cooling("groq", i, retry_after(r, r.text))
            continue
        if r.status_code >= 400:
            mark_cooling("groq", i, 300)
            continue
        return (r.json().get("text") or "").strip()
    return ""


# --- messaging ---

def setup_text() -> str:
    lines = ["Free AI keys (free signup, no card). Add any with:  /keys add <provider> <key>", ""]
    lines.append("  gemini      https://aistudio.google.com/apikey  (main brain + voice)")
    for p in PROVIDERS:
        if not p.recommended:
            continue
        lines.append(f"  {p.name:<11} {p.signup}  — {p.note}")
    lines.append("")
    lines.append("You can add several keys per provider; Atlas rotates them when one runs out.")
    return "\n".join(lines)


def exhausted_notice(gemini_reset: float | None = None) -> str:
    """One friendly line set: what's out, when it resets, which free keys could help."""
    from datetime import datetime
    def fmt(ts: float) -> str:
        return datetime.fromtimestamp(ts).strftime("%H:%M") if ts else "soon"
    parts = ["All my free AI quota is used up right now." if (gemini_reset or configured())
             else "I don't have a working free AI key right now."]
    resets = []
    if gemini_reset:
        resets.append(f"Gemini ~{fmt(gemini_reset)}")
    for name in configured():
        t = next_reset(name)
        if t:
            resets.append(f"{BY_NAME[name].label} ~{fmt(t)}")
    if resets:
        parts.append("Resets (local time): " + ", ".join(resets) + ".")
    missing = [p for p in PROVIDERS if p.recommended and p.name not in configured()]
    if missing:
        parts.append("Add another free key to keep going: " +
                     "; ".join(f"/keys add {p.name} <key> ({p.signup})" for p in missing[:3]) + ".")
    parts.append("PC commands like “open chrome”, “screenshot”, “volume up” still work offline.")
    return " ".join(parts)
