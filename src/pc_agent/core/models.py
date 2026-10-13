"""Model ranking + fast automatic failover across Gemini models.

Policy (latency first):
- 404/NOT_FOUND  -> mark model dead (persisted), skip immediately, no retry.
- 429/RESOURCE_EXHAUSTED -> cooldown honoring retryDelay (persisted), skip, no retry.
- 503/UNAVAILABLE -> at most one quick retry (300 ms), then next model.
- 500/INTERNAL -> next model, no retry.
- Working fallback is sticky for COOLDOWN_SEC; then the preferred model is tried again.
- models.list is never fetched in the request path (background refresh only).
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable

logger = logging.getLogger("pc_agent.models")

EXCLUDE_TEXT = (
    "lite", "tts", "embed", "image", "veo", "live", "audio", "imagen", "aqa",
    "gemma", "learnlm", "robotics", "computer-use", "customtools", "thinking-exp",
)

LIST_TTL_SEC = 3 * 3600
COOLDOWN_SEC = 15 * 60
RETRY_503_DELAY_SEC = 0.3
DEFAULT_429_COOLDOWN_SEC = 60.0
DEAD_TTL_SEC = 24 * 3600  # persisted 404s expire after a day

ALL_FAILED_MSG = (
    "All the Gemini models I can reach are busy or unavailable right now. "
    "Give it a minute and try again — I'll keep checking."
)


class AllModelsFailed(RuntimeError):
    pass


def _code(exc: BaseException) -> int | None:
    for attr in ("code", "status_code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    s = str(exc)
    m = re.search(r"\b(404|429|500|503)\b", s)
    if m:
        return int(m.group(1))
    if "NOT_FOUND" in s or "no longer available" in s:
        return 404
    if "RESOURCE_EXHAUSTED" in s:
        return 429
    if "UNAVAILABLE" in s or "high demand" in s or "overloaded" in s:
        return 503
    if "INTERNAL" in s:
        return 500
    return None


def is_retryable(exc: BaseException) -> bool:
    return _code(exc) in (404, 429, 500, 503)


def retry_delay_sec(exc: BaseException) -> float:
    """Parse retryDelay ('37s' / 'retry in 12.5s') from a 429 error."""
    s = str(exc)
    m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", s) or re.search(
        r"retry in (\d+(?:\.\d+)?)\s*s", s, re.I
    )
    return float(m.group(1)) if m else DEFAULT_429_COOLDOWN_SEC


def _version_key(name: str) -> tuple:
    m = re.search(r"gemini-(\d+)(?:\.(\d+))?", name)
    major = int(m.group(1)) if m else 0
    minor = int(m.group(2) or 0) if m else 0
    stable = 0 if ("preview" in name or "exp" in name) else 1
    latest_alias = 0 if name.endswith("-latest") else 1
    return (major, minor, stable, latest_alias)


def rank_text_models(names: Iterable[str]) -> list[str]:
    """All flash models newest first, then pro models (free-tier unfriendly) last."""
    good = []
    for n in names:
        n = n.replace("models/", "")
        low = n.lower()
        if not low.startswith("gemini-"):
            continue
        if any(x in low for x in EXCLUDE_TEXT):
            continue
        if "flash" not in low and "pro" not in low:
            continue
        good.append(n)
    good = list(dict.fromkeys(good))
    flash = sorted([n for n in good if "flash" in n], key=_version_key, reverse=True)
    pro = sorted([n for n in good if "flash" not in n], key=_version_key, reverse=True)
    return flash + pro


def rank_embedding_models(names: Iterable[str]) -> list[str]:
    out = [n.replace("models/", "") for n in names if "embed" in n.lower()]
    out = list(dict.fromkeys(out))
    return sorted(out, key=lambda n: (("gemini" in n), _version_key(n), n), reverse=True)


class ModelHealth:
    """Shared dead/cooldown state, persisted to disk across restarts."""

    def __init__(self, path: Path | None = None, clock: Callable[[], float] = time.time) -> None:
        self.path = path
        self._clock = clock
        self._lock = threading.Lock()
        self.dead: dict[str, float] = {}      # model -> marked_at
        self.cooling: dict[str, float] = {}   # model -> until
        self._load()

    def _load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8-sig"))
            now = self._clock()
            self.dead = {k: v for k, v in (data.get("dead") or {}).items() if now - v < DEAD_TTL_SEC}
            self.cooling = {k: v for k, v in (data.get("cooling") or {}).items() if v > now}
        except Exception as exc:
            logger.debug("model health load failed: %s", exc)

    def _save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"dead": self.dead, "cooling": self.cooling}), encoding="utf-8")
        except Exception as exc:
            logger.debug("model health save failed: %s", exc)

    def usable(self, model: str) -> bool:
        with self._lock:
            if model in self.dead:
                return False
            until = self.cooling.get(model)
            if until and until > self._clock():
                return False
            if until:
                self.cooling.pop(model, None)
            return True

    def mark_dead(self, model: str) -> None:
        with self._lock:
            self.dead[model] = self._clock()
            self._save()

    def mark_cooling(self, model: str, seconds: float) -> None:
        with self._lock:
            self.cooling[model] = self._clock() + max(1.0, seconds)
            self._save()

    def clear(self, model: str) -> None:
        with self._lock:
            changed = self.dead.pop(model, None) is not None or self.cooling.pop(model, None) is not None
            if changed:
                self._save()


class ModelRouter:
    """Ranked candidates + fast failover. fn(model) does the API call."""

    def __init__(
        self,
        kind: str,
        candidates: list[str],
        preferred: str | None = None,
        on_switch: Callable[[str, str], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
        health: ModelHealth | None = None,
    ) -> None:
        self.kind = kind
        self._lock = threading.RLock()
        self.preferred = preferred
        self.candidates = self._order(candidates, preferred)
        self.current: str | None = self.candidates[0] if self.candidates else preferred
        self._fallback_since: float | None = None
        self.on_switch = on_switch
        self._sleep = sleep
        self._clock = clock
        self.health = health or ModelHealth(None, clock=clock)
        self.last_note = ""
        self.last_attempts: list[tuple[str, str, float]] = []  # (model, outcome, secs)

    @staticmethod
    def _order(cands: list[str], preferred: str | None) -> list[str]:
        c = list(dict.fromkeys(cands))
        if preferred:
            c = [preferred] + [x for x in c if x != preferred]
        return c

    def set_candidates(self, cands: list[str]) -> None:
        with self._lock:
            self.candidates = self._order(cands, self.preferred)
            if self.current not in self.candidates and self.candidates:
                self.current = self.candidates[0]

    def _start_order(self) -> list[str]:
        with self._lock:
            if self._fallback_since is not None and self._clock() - self._fallback_since >= COOLDOWN_SEC:
                self._fallback_since = None
                self.current = self.candidates[0] if self.candidates else self.current
            cur = self.current
            if cur and cur in self.candidates:
                i = self.candidates.index(cur)
                # Sticky current first, then the rest in rank order (not wrapped)
                order = [cur] + [m for m in self.candidates if m != cur]
            else:
                order = list(self.candidates)
        usable = [m for m in order if self.health.usable(m)]
        return usable

    def call(self, fn: Callable[[str], Any]) -> Any:
        order = self._start_order()
        self.last_attempts = []
        if not order:
            self.last_note = ALL_FAILED_MSG
            raise AllModelsFailed(f"no usable {self.kind} models")
        last: BaseException | None = None
        for model in order:
            tries = 0
            while True:
                t0 = self._clock()
                try:
                    result = fn(model)
                    self.last_attempts.append((model, "ok", self._clock() - t0))
                    self._mark_ok(model)
                    return result
                except Exception as exc:
                    code = _code(exc)
                    dt = self._clock() - t0
                    last = exc
                    if code not in (404, 429, 500, 503):
                        self.last_attempts.append((model, f"error:{type(exc).__name__}", dt))
                        raise
                    self.last_attempts.append((model, str(code), dt))
                    if code == 404:
                        self.health.mark_dead(model)
                        break
                    if code == 429:
                        self.health.mark_cooling(model, retry_delay_sec(exc))
                        break
                    if code == 503 and tries == 0:
                        tries += 1
                        self._sleep(RETRY_503_DELAY_SEC)
                        continue
                    if code == 503:
                        # brief cooldown so the next request doesn't hit it again
                        self.health.mark_cooling(model, 30)
                    break
        self.last_note = ALL_FAILED_MSG
        raise AllModelsFailed(str(last))

    def _mark_ok(self, model: str) -> None:
        with self._lock:
            prev = self.current
            self.current = model
            top = next((m for m in self.candidates if self.health.usable(m)), model)
            if model != (self.candidates[0] if self.candidates else model):
                if self._fallback_since is None or prev != model:
                    self._fallback_since = self._clock()
            else:
                self._fallback_since = None
            self.last_note = f"Using {model}"
            if prev != model and self.on_switch:
                try:
                    self.on_switch(prev or "", model)
                except Exception:
                    pass


class ModelCatalog:
    """models.list cache. Request path only ever reads the cache."""

    def __init__(self, client, clock: Callable[[], float] = time.time, cache_path: Path | None = None) -> None:
        self.client = client
        self._clock = clock
        self._at = 0.0
        self._names: list[str] = []
        self._lock = threading.Lock()
        self.cache_path = cache_path
        self._load_cache()

    def _load_cache(self) -> None:
        if not self.cache_path or not self.cache_path.exists():
            return
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8-sig"))
            self._names = list(data.get("names") or [])
            self._at = float(data.get("at") or 0)
        except Exception:
            pass

    def stale(self) -> bool:
        return not self._names or self._clock() - self._at >= LIST_TTL_SEC

    def refresh(self) -> list[str]:
        """Blocking fetch — call from a background thread only."""
        names: list[str] = []
        try:
            for m in self.client.models.list():
                name = (getattr(m, "name", "") or "").replace("models/", "")
                actions = str(
                    getattr(m, "supported_actions", None)
                    or getattr(m, "supported_generation_methods", None)
                    or ""
                )
                if actions and "embed" in name.lower():
                    if "embedContent" not in actions:
                        continue
                elif actions and "generateContent" not in actions:
                    continue
                names.append(name)
        except Exception as exc:
            logger.warning("models.list failed: %s", exc)
        if names:
            with self._lock:
                self._names = names
                self._at = self._clock()
            if self.cache_path:
                try:
                    self.cache_path.write_text(json.dumps({"names": names, "at": self._at}), encoding="utf-8")
                except Exception:
                    pass
        return self._names

    def refresh_async(self, on_done: Callable[[], None] | None = None) -> threading.Thread:
        def _run() -> None:
            self.refresh()
            if on_done:
                try:
                    on_done()
                except Exception:
                    logger.exception("catalog on_done failed")
        t = threading.Thread(target=_run, name="pc-agent-models", daemon=True)
        t.start()
        return t

    def names(self, force: bool = False) -> list[str]:
        """Cached names only. Never fetches unless force=True (background/tests)."""
        if force:
            return self.refresh()
        return self._names

    def text_models(self, fallback: list[str]) -> list[str]:
        ranked = rank_text_models(self.names())
        return ranked or list(fallback)

    def embedding_models(self, fallback: list[str]) -> list[str]:
        ranked = rank_embedding_models(self.names())
        return ranked or list(fallback)
