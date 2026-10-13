"""Friendly free-tier quota messaging ($0 — no billing pressure)."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone


def parse_retry_seconds(exc_or_text) -> float | None:
    s = str(exc_or_text)
    m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", s) or re.search(
        r"retry in (\d+(?:\.\d+)?)s", s, re.I
    )
    if m:
        return float(m.group(1))
    m = re.search(r"retry in\s+(\d+)h(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?", s, re.I)
    if m:
        h = int(m.group(1))
        mi = int(m.group(2) or 0)
        sec = float(m.group(3) or 0)
        return h * 3600 + mi * 60 + sec
    m = re.search(r"(\d+)h(\d+)m", s)
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60
    return None


def is_quota_exhausted(exc_or_text) -> bool:
    s = str(exc_or_text)
    return (
        "RESOURCE_EXHAUSTED" in s
        or "429" in s
        or "quota" in s.lower()
        or "free_tier" in s.lower()
        or "free tier" in s.lower()
        or "All the Gemini models" in s
    )


def _fmt_local(seconds: float, local_tz=None) -> str:
    if local_tz is None:
        local_tz = datetime.now().astimezone().tzinfo
    reset_at = datetime.now(timezone.utc).astimezone(local_tz) + timedelta(seconds=max(0, seconds))
    return reset_at.strftime("%I:%M %p").lstrip("0")


def format_quota_notice(exc_or_text, *, local_tz=None, model_resets: dict[str, float] | None = None) -> str:
    """One-line notice with local reset time. No billing mention ($0 design)."""
    secs = parse_retry_seconds(exc_or_text)
    if secs is None:
        secs = 3600
    when = _fmt_local(secs, local_tz)
    parts = [
        f"Free Gemini quota is used up until about {when} local.",
        "I'll keep using offline smarts (open apps, folders, screenshots) and free local voice.",
    ]
    if model_resets:
        bits = []
        for model, s in sorted(model_resets.items(), key=lambda kv: kv[1])[:4]:
            bits.append(f"{model} ~{_fmt_local(s, local_tz)}")
        if bits:
            parts.append("Resets: " + ", ".join(bits) + ".")
    return " ".join(parts)


def model_reset_map(health) -> dict[str, float]:
    """Seconds-until-ready for cooling models from ModelHealth."""
    out: dict[str, float] = {}
    try:
        now = health._clock()
        for model, until in (health.cooling or {}).items():
            remaining = until - now
            if remaining > 0:
                out[model] = remaining
    except Exception:
        pass
    return out
