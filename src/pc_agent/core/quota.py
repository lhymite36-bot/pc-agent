"""Friendly free-tier quota messaging."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone


def parse_retry_seconds(exc_or_text) -> float | None:
    s = str(exc_or_text)
    m = re.search(r"retry in (\d+(?:\.\d+)?)s", s, re.I) or re.search(
        r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", s
    )
    if m:
        return float(m.group(1))
    # e.g. "7h26m14s" / "Please retry in 7h26m"
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
    )


def format_quota_notice(exc_or_text, *, local_tz=None) -> str:
    """One-line notice with local reset time + billing hint."""
    secs = parse_retry_seconds(exc_or_text)
    if secs is None:
        secs = 3600
    if local_tz is None:
        local_tz = datetime.now().astimezone().tzinfo
    reset_at = datetime.now(timezone.utc).astimezone(local_tz) + timedelta(seconds=secs)
    when = reset_at.strftime("%I:%M %p").lstrip("0") + " local"
    return (
        f"Free Gemini quota is used up until about {when}. "
        f"Replies will use offline voice for now — turn on billing in Google AI Studio to keep going."
    )
